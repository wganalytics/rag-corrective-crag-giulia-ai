import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .evaluator import RelevanceAuditor
from .reformulator import QueryReformulator
from .validator import HallucinationValidator

load_dotenv()

# Configurações
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", "llama3.2:3b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3.2:3b") # Sugerido 7b em prod
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")

# Caminhos
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CorrectiveRAG")

class CorrectiveRAG:
    def __init__(self):
        # Motores especializados
        self.auditor = RelevanceAuditor(AUDITOR_MODEL, OLLAMA_HOST)
        self.reformulator = QueryReformulator(AUDITOR_MODEL, OLLAMA_HOST)
        self.validator = HallucinationValidator(AUDITOR_MODEL, OLLAMA_HOST)
        
        # Gerador Final
        self.generator_llm = ChatOllama(model=GENERATOR_MODEL, temperature=0, base_url=OLLAMA_HOST)
        
        # Vector Store Placeholder
        self._vs = None

    def _get_vs(self):
        if self._vs is None:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            self._vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return self._vs

    @property
    def uploads_dir(self):
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        return UPLOADS_DIR

    def process_pdf(self, path: str):
        logger.info(f"Processando PDF: {path}")
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        
        # Chunking Jurídico (Regex poderia ser melhor, mas usaremos recursivo por enquanto)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=80,
            separators=["\nArt.", "\n§", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)
        
        vs = self._get_vs()
        vs.add_documents(chunks)
        logger.info(f"Ingestão concluída: {len(chunks)} chunks adicionados.")
        return len(chunks)

    def query(self, question: str, max_retries: int = 2) -> Dict:
        vs = self._get_vs()
        current_query = question
        history = []

        for attempt in range(max_retries + 1):
            logger.info(f"Tentativa {attempt + 1} | Query: {current_query}")
            
            # 1. Retrieval
            docs = vs.similarity_search(current_query, k=5)
            context = "\n---\n".join([d.page_content for d in docs])
            
            # 2. Audit (Self-Reflection)
            critique = self.auditor.audit(question, context)
            logger.info(f"Resultado da Auditoria: {critique}")
            
            history.append({
                "attempt": attempt + 1,
                "query": current_query,
                "metrics": critique
            })

            if critique.get("action") == "generate" or attempt == max_retries:
                # 3. Generation
                if attempt == max_retries and critique.get("relevance", 0) < 0.5:
                    return {
                        "answer": "Sinto muito, mas após múltiplas tentativas, não encontrei base documental suficiente nos arquivos jurídicos para responder sua pergunta com precisão.",
                        "quality": "fail",
                        "metrics": critique,
                        "history": history
                    }

                prompt = PromptTemplate.from_template("""
                Você é um Assistente Jurídico especializado. Responda à pergunta baseando-se ESTRITAMENTE no contexto fornecido.
                Se não houver base legal no contexto, informe que a informação não foi localizada nos documentos.
                
                Contexto:
                {context}
                
                Pergunta: {question}
                
                Resposta Técnica (Citar artigos/cláusulas se disponíveis):""")
                
                final_answer = self.generator_llm.invoke(prompt.format(context=context, question=question)).content
                
                # 4. Hallucination Validation
                logger.info("Iniciando validação de alucinação...")
                validation_results = self.validator.validate_answer(final_answer, context)
                support_score = sum(1 for v in validation_results if v.get("has_support")) / len(validation_results) if validation_results else 1.0
                
                return {
                    "answer": final_answer,
                    "quality": "verified" if (critique.get("relevance", 0) > 0.7 and support_score > 0.8) else "uncertain",
                    "metrics": {**critique, "support_score": support_score},
                    "validation": validation_results,
                    "history": history,
                    "sources": list(set([d.metadata.get("source", "N/A") for d in docs]))
                }

            # 5. Reformulation (Retry)
            if attempt == 0:
                current_query = self.reformulator.expand(question)
            elif attempt == 1:
                current_query = self.reformulator.rewrite(question)

        return {"error": "Max retries exceeded"}