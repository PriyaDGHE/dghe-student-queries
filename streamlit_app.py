```python
# streamlit_app.py — DGHE Student Queries Bot (MVP)
# Runs on Streamlit Community Cloud. Set OPENAI_API_KEY in Settings → Secrets.

import os
import io
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# OpenAI SDK (>=1.40.0)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Page config
st.set_page_config(page_title="DGHE Student Queries Bot", page_icon="🎓", layout="centered")
st.title("🎓 DGHE Student Queries Bot")
st.caption("Answers routine questions fast. Escalates the rest to humans.")

# ---------------- API key handling (Secrets → OPENAI_API_KEY)
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    pass
OPENAI_API_KEY = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

if OpenAI is None:
    st.error("OpenAI SDK not found. Check requirements.txt includes `openai>=1.40.0`.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Small status indicator so you know if keys are wired
if OPENAI_API_KEY:
    st.success("API key detected. Responses are live.")
else:
    st.warning("No API key found. Add OPENAI_API_KEY in Settings → Secrets to enable answers.")

SYSTEM_PROMPT = (
    "You are a friendly, concise student support assistant for a UK Higher Education provider (DGHE). "
    "Tone: supportive, clear, professional, encouraging. "
    "Always cite the source section titles or filenames you used in square brackets at the end, e.g., "
    "[Extenuating Circumstances Policy, p.3]. "
    "If the answer is uncertain or policy isn’t found in context, say so and propose next steps or escalate to a human. "
    "If the question is about deadlines or extensions, remind that official dates on Moodle prevail."
)

# ---------------- Helpers

@st.cache_data(show_spinner=False)
def load_faqs_from_csv(file) -> pd.DataFrame:
    """Load a FAQs CSV with columns (question,answer,category). If file is None, try repo file."""
    if file is None:
        try:
            df = pd.read_csv("dghe_kb_faqs.csv")
            return df[["question", "answer", "category"]]
        except Exception:
            # last-resort tiny sample so UI still loads
            df = pd.DataFrame(
                [
                    {
                        "question": "How do I request an extension?",
                        "answer": "Submit a Mitigating Circumstances application before the deadline via the Student Portal.",
                        "category": "Assessment",
                    }
                ]
            )
            return df
    df = pd.read_csv(file)
    # case-insensitive column fix
    cols_lower = {c.lower(): c for c in df.columns}
    for needed in ["question", "answer", "category"]:
        if needed not in cols_lower:
            raise ValueError("CSV must include columns: question, answer, category")
    df = df.rename(
        columns={
            cols_lower["question"]: "question",
            cols_lower["answer"]: "answer",
            cols_lower["category"]: "category",
        }
    )
    return df[["question", "answer", "category"]]


@st.cache_data(show_spinner=False)
def pdfs_to_docs(files: List[io.BytesIO]) -> List[Dict]:
    """Turn uploaded PDFs into page-level text docs."""
    docs: List[Dict] = []
    if not files:
        return docs
    for f in files:
        try:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        {"title": getattr(f, "name", "Policy.pdf"), "page": i + 1, "text": text}
                    )
        except Exception:
            continue  # ignore unreadable PDFs; keep going
    return docs


@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    """Get embeddings; fall back to deterministic random vectors if no API key (demo mode)."""
    if not OPENAI_API_KEY or client is None:
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 256)).astype(np.float32)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


@st.cache_data(show_spinner=False)
def build_kb(faq_df: pd.DataFrame, policy_docs: List[Dict]):
    """Build chunk list + embeddings from FAQs and policy PDFs."""
    chunks: List[str] = []
    sources: List[Dict] = []

    # FAQs: treat each Q/A as a chunk
    for _, row in faq_df.iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        cat = str(row.get("category", "General")).strip()
        chunks.append(f"Q: {q}\nA: {a}")
        sources.append({"source": f"FAQ — {cat}", "detail": q[:80] + ("…" if len(q) > 80 else "")})

    # Policies: chunk by paragraph-ish blocks
    for d in policy_docs:
        title = d["title"]
        page = d["page"]
        for para in d["text"].split("\n\n"):
            p = para.strip()
            if len(p) < 200:
                continue  # skip tiny fragments
            chunks.append(p)
            sources.append({"source": title, "detail": f"p.{page}"})

    if not chunks:
        chunks = ["No knowledge uploaded yet. Answer only with general guidance and suggest escalation."]
        sources = [{"source": "System", "detail": "empty KB"}]

    embs = embed_texts(chunks)
    return chunks, sources, embs


def retrieve(query: str, chunks: List[str], sources: List[Dict], embs: np.ndarray, k: int = 6):
    """Simple cosine-similarity retrieval."""
    q_emb = embed_texts([query])
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = np.argsort(-sims)[:k]
    ctx = []
    used = []
    for i in idxs:
        ctx.append(chunks[i])
        used.append(sources[i])
    return "\n\n---\n\n".join(ctx), used


def generate_answer(query: str, context: str, used_sources: List[Dict]) -> str:
    """Call the LLM with retrieved context; append lightweight citations."""
    if not OPENAI_API_KEY or client is None:
        return "(Demo mode) Add OPENAI_API_KEY in Streamlit Secrets to enable answers."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Relevant context from policies/FAQs:\n{context}\n\n"
                "Write a helpful answer. If context is weak or conflicting, say you’re unsure and offer next steps."
            ),
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    ans = resp.choices[0].message.content

    if used_sources:
        citations = ", ".join([f"{u['source']} {u['detail']}" for u in used_sources[:3]])
        ans += f"\n\n[Sources: {citations}]"
    return ans


# ---------------- UI: Knowledge upload / build

with st.expander("Upload knowledge (admin)", expanded=True):
    faqs_file = st.file_uploader("Upload FAQs CSV (question,answer,category)", type=["csv"], key="faqs")
    policies_files = st.file_uploader("Upload policy PDFs (optional)", type=["pdf"], accept_multiple_files=True, key="pdfs")

    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False

    if st.button("Build / Refresh Knowledge Base", type="primary"):
        try:
            df = load_faqs_from_csv(faqs_file)
        except Exception as e:
            st.error(f"Could not read FAQ CSV: {e}")
            st.stop()
        docs = pdfs_to_docs(policies_files) if policies_files else []
        chunks, sources, embs = build_kb(df, docs)
        st.session_state.kb = {"chunks": chunks, "sources": sources, "embs": embs}
        st.session_state.kb_ready = True
        st.success(f"Knowledge base ready with {len(chunks)} chunks.")

if not st.session_state.get("kb_ready"):
    st.info(
        "Tip: If you don’t upload a CSV, the app will try to use **dghe_kb_faqs.csv** in the repo. "
        "You can also add policy PDFs for better citations, then click **Build / Refresh Knowledge Base**."
    )

st.divider()

# ---------------- Sidebar: escalation template
with st.sidebar:
    st.subheader("Escalate to human")
    student_email = st.text_input("Student email (optional)")
    if st.button("Copy email draft"):
        last_q = ""
        for m in reversed(st.session_state.get("history", [])):
            if m["role"] == "user":
                last_q = m["content"]
                break
        draft = f"""Subject: Support with {last_q or "your recent question"}

Dear Student,

Thanks for your message about "{last_q or "your recent question"}". I’d like to ensure you get the correct, course-specific guidance. I’m escalating this to our programme team and will get back to you shortly. If urgent, please include your cohort, module, and a screenshot of the issue.

Kind regards,
DGHE Student Support
"""
        st.code(draft, language="text")

# ---------------- Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question (e.g., 'How do I request an extension?')")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            kb = st.session_state.get("kb")
            if kb:
                ctx, used = retrieve(user_query, kb["chunks"], kb["sources"], kb["embs"], k=6)
            else:
                # Auto-build from repo CSV if available
                df = load_faqs_from_csv(None)
                chunks, sources, embs = build_kb(df, [])
                ctx, used = retrieve(user_query, chunks, sources, embs, k=6)

            answer = generate_answer(user_query, ctx, used)
            st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})

st.caption(
    "Privacy: This demo stores data only in your Streamlit session state. "
    "For production, add authentication, logging policy, and PII redaction."
)
```






