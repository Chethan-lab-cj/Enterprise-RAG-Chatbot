import os
import re
from pathlib import Path

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="NLP Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("NLP Knowledge Chatbot")
st.caption("AI-powered document question-answering system")

PDF_PATH = Path(__file__).parent / "2026 NLP_ TLEP.docx.pdf"


@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured. Add GOOGLE_API_KEY "
            "in the app's Streamlit Secrets settings."
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.name}. "
            "Upload the PDF alongside app.py."
        )

    reader = PdfReader(str(PDF_PATH))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("No readable text could be extracted from the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks).tolist()

    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="nlp_tlep")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": PDF_PATH.name, "chunk": i + 1}
            for i in range(len(chunks))
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0
    )

    return embedding_model, collection, llm


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about NLP...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Loading resources and searching the document..."):
                embedding_model, collection, llm = load_resources()

                query_embedding = embedding_model.encode([question]).tolist()
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(4, collection.count()),
                    include=["documents", "metadatas"]
                )

            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if not docs:
                answer = "I couldn't find this information in the provided document."
                sources = []
            else:
                context = "\n\n".join(docs)
                prompt = f"""
You are an assistant answering questions from NLP course documents.

Answer the user's specific question directly. Give enough detail to answer
it properly. Do not force every answer into a fixed number of points or
sentences. Use only the information in the context. Do not invent facts
or add unrelated information. If the answer is not available in the context,
say that it could not be found in the document.

Context:
{context}

Question:
{question}

Answer:
"""
                response = llm.invoke(prompt)
                content = response.content

                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                else:
                    answer = str(content)

                sources = list(dict.fromkeys(
                    f"{meta.get('source', 'Unknown')} "
                    f"(Chunk {meta.get('chunk', '?')})"
                    for meta in metas
                ))

            st.markdown(answer)
            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.write(source)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Unable to generate an answer: {e}")
