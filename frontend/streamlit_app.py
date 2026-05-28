"""
PRJ-04: Corrective RAG — Streamlit Interface
"""

import streamlit as st
import requests

API_URL = "http://localhost:8003"

st.set_page_config(page_title="Corrective RAG", page_icon="🔍")
st.title("🔍 Corrective RAG")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Configuração")
    max_retries = st.slider("Max retries", 0, 3, 2)
    
    st.divider()
    st.header("Upload PDF")
    uploaded = st.file_uploader("Enviar documento", type=["pdf"])
    if st.button("Processar") and uploaded:
        with st.spinner("Processando..."):
            files = {"file": uploaded.getvalue()}
            # Simplified upload

st.divider()

query = st.text_input("Pergunta jurídica:", key="query_input")
if st.button("Executar") and query:
    with st.spinner("Analisando..."):
        resp = requests.post(f"{API_URL}/query", json={"question": query, "max_retries": max_retries})
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.history.append({
                "q": query,
                "a": result["answer"],
                "quality": result["quality"],
                "attempts": result["attempts"]
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Qualidade", result["quality"].upper())
            with col2:
                st.metric("Tentativas", result["attempts"])
            
            st.text_area("Resposta:", result["answer"], height=200)

st.divider()
st.subheader("Histórico")
for h in reversed(st.session_state.history[-5:]):
    with st.expander(f"Q: {h['q'][:50]}... | {h['quality']}"):
        st.write(h["a"])