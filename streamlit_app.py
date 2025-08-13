python -m venv .venv
. .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
streamlit run streamlit_app.py
