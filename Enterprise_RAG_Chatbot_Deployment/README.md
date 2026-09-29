# NLP Knowledge Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot for asking questions about NLP course material.

## Files
- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `2026 NLP_ TLEP.docx.pdf`: Add your course PDF beside `app.py`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure the Gemini API key as `GOOGLE_API_KEY` in the environment or in Streamlit Secrets. Do not commit API keys.

The app builds an in-memory Chroma collection from the PDF at runtime. The first load can take a while because it loads the embedding model and processes the document.
