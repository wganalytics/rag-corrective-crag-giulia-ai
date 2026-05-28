import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Adiciona o diretório base do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# ──────────────────────────────────────────────────────────────
# ULTRA-FAST HERMETIC LANGCHAIN & GRPC MOCKS (sys.modules)
# ──────────────────────────────────────────────────────────────
mock_loaders = MagicMock()
sys.modules['langchain_community.document_loaders'] = mock_loaders

mock_splitters = MagicMock()
sys.modules['langchain_text_splitters'] = mock_splitters

mock_vectorstores = MagicMock()
sys.modules['langchain_community.vectorstores'] = mock_vectorstores

mock_ollama = MagicMock()
sys.modules['langchain_ollama'] = mock_ollama

mock_prompts = MagicMock()
mock_prompts.PromptTemplate = MagicMock()
sys.modules['langchain_core.prompts'] = mock_prompts

from langchain_core.documents import Document

# Agora importamos os módulos de forma rápida e segura!
from core.evaluator import RelevanceAuditor
from core.reformulator import QueryReformulator
from core.validator import HallucinationValidator
from core.corrective_engine import CorrectiveRAG

@pytest.fixture
def mock_evaluator():
    auditor = RelevanceAuditor("llama3.2:3b", "http://localhost:11434")
    auditor.llm = MagicMock()
    return auditor

@pytest.fixture
def mock_reformulator():
    reformulator = QueryReformulator("llama3.2:3b", "http://localhost:11434")
    reformulator.llm = MagicMock()
    return reformulator

@pytest.fixture
def mock_validator():
    validator = HallucinationValidator("llama3.2:3b", "http://localhost:11434")
    validator.llm = MagicMock()
    return validator

@pytest.fixture
def mock_corrective_rag():
    with patch('core.corrective_engine.OllamaEmbeddings'), \
         patch('core.corrective_engine.Chroma') as mock_chroma:
        
        vs = MagicMock()
        mock_chroma.return_value = vs
        
        engine = CorrectiveRAG()
        engine.generator_llm = MagicMock()
        engine.auditor.llm = MagicMock()
        engine.reformulator.llm = MagicMock()
        engine.validator.llm = MagicMock()
        yield engine, vs

# ──────────────────────────────────────────────────────────────
# CASOS DE TESTE TDD — PRJ-04 CORRECTIVE RAG
# ──────────────────────────────────────────────────────────────

def test_auditor_high_relevance(mock_evaluator):
    """Verifica se o auditor aprova o contexto para geração quando há alta relevância."""
    mock_response = MagicMock()
    mock_response.content = '{"relevance": 0.85, "completeness": 0.8, "support": 0.9, "is_approved": true, "action": "generate", "reasoning": "Contexto completo"}'
    mock_evaluator.llm.invoke.return_value = mock_response
    
    critique = mock_evaluator.audit("Pergunta sobre contrato", "O contrato diz que...")
    
    assert critique["action"] == "generate"
    assert critique["relevance"] == 0.85

def test_auditor_low_relevance(mock_evaluator):
    """Verifica se o auditor aciona a reformulação da query quando a relevância é baixa."""
    mock_response = MagicMock()
    mock_response.content = '{"relevance": 0.2, "completeness": 0.1, "support": 0.1, "is_approved": false, "action": "retry", "reasoning": "Irrelevante"}'
    mock_evaluator.llm.invoke.return_value = mock_response
    
    critique = mock_evaluator.audit("Pergunta sobre contrato", "O tempo está ensolarado")
    
    assert critique["action"] == "retry"
    assert critique["relevance"] == 0.2

def test_reformulator_expand(mock_reformulator):
    """Garante que a expansão jurídica torna a query mais abrangente e técnica."""
    mock_response = MagicMock()
    mock_response.content = "prazo de rescisão contratual e distrato de locação"
    mock_reformulator.llm.invoke.return_value = mock_response
    
    expanded = mock_reformulator.expand("acabar contrato")
    
    assert len(expanded) > len("acabar contrato")
    assert "rescisão" in expanded or "contratual" in expanded

def test_hallucination_validator(mock_validator):
    """Verifica se o validador atesta corretamente quando o sistema gera uma resposta com suporte literal."""
    mock_response = MagicMock()
    mock_response.content = '{"has_support": true, "source_fragment": "Art. 12", "explanation": "Suportado pelo documento", "confidence": 0.95}'
    mock_validator.llm.invoke.return_value = mock_response
    
    validation = mock_validator.validate_answer("O Artigo 12 prevê rescisão em 30 dias.", "Art. 12: Rescisão em 30 dias.")
    
    assert len(validation) > 0
    assert validation[0]["has_support"] is True

def test_max_retries_fail(mock_corrective_rag):
    """Garante que após falharem todos os retries por falta de relevância, o sistema recusa responder graciosamente."""
    engine, vs = mock_corrective_rag
    
    # Mock do similarity search retornando documentos irrelevantes
    doc = Document(page_content="Texto qualquer", metadata={"source": "contrato.pdf"})
    vs.similarity_search.return_value = [doc]
    
    # Mock do auditor sempre retornando action="retry" e score baixo
    engine.auditor.llm.invoke.return_value.content = '{"relevance": 0.1, "completeness": 0.1, "support": 0.1, "is_approved": false, "action": "retry"}'
    
    # Mock do reformulator
    engine.reformulator.llm.invoke.return_value.content = "Query reformulada"
    
    response = engine.query("Qual o prazo?", max_retries=2)
    
    assert response["quality"] == "fail"
    assert "não encontrei base documental suficiente" in response["answer"]

def test_quality_verified(mock_corrective_rag):
    """Garante que o pipeline retorna qualidade 'verified' com alta relevância e validação sem alucinação."""
    engine, vs = mock_corrective_rag
    
    doc = Document(page_content="Art. 12: Prazo de 30 dias.", metadata={"source": "contrato.pdf"})
    vs.similarity_search.return_value = [doc]
    
    # Mock auditor retornando generate
    engine.auditor.llm.invoke.return_value.content = '{"relevance": 0.9, "completeness": 0.9, "support": 0.9, "is_approved": true, "action": "generate"}'
    
    # Mock generator
    engine.generator_llm.invoke.return_value = MagicMock(content="O Artigo 12 estabelece o prazo de 30 dias.")
    
    # Mock validator
    engine.validator.llm.invoke.return_value.content = '{"has_support": true, "source_fragment": "Art. 12", "explanation": "Verificado", "confidence": 0.99}'
    
    response = engine.query("Qual o prazo de rescisão?", max_retries=2)
    
    assert response["quality"] == "verified"
    assert "O Artigo 12" in response["answer"]
    assert response["sources"] == ["contrato.pdf"]

@patch('core.corrective_engine.PyMuPDFLoader')
@patch('core.corrective_engine.RecursiveCharacterTextSplitter')
def test_pdf_chunking_juridico(mock_splitter_class, mock_loader_class, mock_corrective_rag):
    """Verifica se o chunking jurídico é executado com os separadores corretos (Art. e §)."""
    engine, vs = mock_corrective_rag
    
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="Art. 1: Caput.\n§ 1: Parágrafo.")]
    mock_loader_class.return_value = mock_loader
    
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [Document(page_content="Art. 1: Caput."), Document(page_content="§ 1: Parágrafo.")]
    mock_splitter_class.return_value = mock_splitter
    
    num_chunks = engine.process_pdf("contrato.pdf")
    
    assert num_chunks == 2
    mock_splitter_class.assert_called_once_with(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\nArt.", "\n§", "\n\n", "\n", " "]
    )
