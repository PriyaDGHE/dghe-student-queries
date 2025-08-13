# 1) Create a folder and put the 3 files below inside it
#    - streamlit_app.py
#    - requirements.txt
#    - sample_faqs.csv

# 2) Create a virtual environment (optional but recommended)
python -m venv .venv
. .venv/bin/activate  # on Windows: .\.venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set your OpenAI API key for this session
#    (get one from platform.openai.com — personal account is fine)
export OPENAI_API_KEY=sk-...   # on Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

# 5) Run the app
streamlit run streamlit_app.py
