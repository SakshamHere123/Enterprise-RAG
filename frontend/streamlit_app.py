"""
Streamlit frontend — a chat-style UI for the RAG system.
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Enterprise Knowledge Worker", page_icon="🧠", layout="centered")

st.title("Enterprise Knowledge Worker")
st.caption("Ask questions about your internal documents — answers are grounded in retrieved context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    st.write("Add files to `data/raw/`, then click below to index them.")
    if st.button("🔄 Reindex documents"):
        with st.spinner("Embedding documents and building FAISS index..."):
            try:
                resp = requests.post(f"{API_URL}/reindex", timeout=300)
                if resp.status_code == 200:
                    st.success("Index rebuilt successfully!")
                else:
                    st.error(f"Failed: {resp.json().get('detail')}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is `app.py` running on port 8000?")

    st.divider()
    st.caption("Stack: LangChain · FAISS · OpenAI · FastAPI · Streamlit")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f" Sources: {', '.join(msg['sources'])}")

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query", json={"question": question}, timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data["sources"]:
                        st.caption(f"📚 Sources: {', '.join(data['sources'])}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                    })
                else:
                    error_msg = resp.json().get("detail", "Unknown error")
                    st.error(error_msg)
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Make sure `app.py` is running (`python app.py`).")