import streamlit as st
import requests
import json
import os
from typing import Dict, List

st.set_page_config(page_title="PRJ-04: Auditor Jurídico CRAG", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8003")

st.title("⚖️ Auditor Jurídico - Corrective RAG")
st.markdown("---")

# Sidebar - Upload e Configurações
with st.sidebar:
    st.header("🗂️ Gestão de Documentos")
    uploaded_file = st.file_uploader("Upload de PDF Jurídico", type="pdf")
    if uploaded_file:
        if st.button("Indexar Documento"):
            with st.spinner("Processando..."):
                files = {"file": uploaded_file.getvalue()}
                resp = requests.post(f"{API_URL}/upload_pdf", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")})
                if resp.status_code == 201:
                    st.success(f"Indexado: {resp.json()['chunks']} chunks.")
                else:
                    st.error("Erro no upload.")

    st.header("⚙️ Configurações")
    max_retries = st.slider("Máximo de Retries", 0, 5, 2)

# Main Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "trace" in message:
            with st.expander("🔍 Thought Trace & Metrics"):
                st.json(message["trace"])

if prompt := st.chat_input("Qual sua dúvida jurídica?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Auditando base técnica..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "max_retries": max_retries}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    
                    # Status de Qualidade
                    quality = data["quality"]
                    if quality == "verified":
                        st.info("✅ Resposta validada e suportada pelo contexto.")
                    elif quality == "uncertain":
                        st.warning("⚠️ Resposta com baixa confiança documental.")
                    else:
                        st.error("❌ Falha na localização de base legal.")

                    st.markdown(answer)
                    
                    # Trace
                    trace = {
                        "metrics": data["metrics"],
                        "history": data["history"],
                        "validation": data.get("validation", []),
                        "sources": data.get("sources", [])
                    }
                    with st.expander("🔍 Thought Trace & Metrics"):
                        st.json(trace)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "trace": trace
                    })
                else:
                    st.error(f"Erro na API: {response.text}")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")
