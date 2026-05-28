from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

EXPANSION_PROMPT = """Como um especialista em Direito, reescreva a pergunta do usuário para torná-la mais técnica e abrangente, facilitando a busca em bases de dados jurídicas (contratos, leis, jurisprudência).

Use sinônimos jurídicos (ex: em vez de 'acabar contrato', use 'rescisão' ou 'distrato').
Adicione termos correlatos que podem estar presentes em leis brasileiras.

Pergunta Original: {question}

Pergunta Expandida:"""

REWRITING_PROMPT = """Analise por que a busca anterior falhou e reescreva a pergunta de forma totalmente diferente, focando no núcleo do problema jurídico.

Query original: {question}
Motivo da falha: Contexto recuperado era irrelevante ou falava de outro tema.

Nova Query Técnica:"""

class QueryReformulator:
    def __init__(self, model_name: str, base_url: str):
        self.llm = ChatOllama(model=model_name, temperature=0.7, base_url=base_url)
        self.expansion_prompt = PromptTemplate.from_template(EXPANSION_PROMPT)
        self.rewriting_prompt = PromptTemplate.from_template(REWRITING_PROMPT)

    def expand(self, question: str) -> str:
        """Tentativa 2: Expansão com sinônimos técnicos."""
        response = self.llm.invoke(self.expansion_prompt.format(question=question))
        return response.content.strip()

    def rewrite(self, question: str) -> str:
        """Tentativa 3: Reescrita completa por falha de contexto."""
        response = self.llm.invoke(self.rewriting_prompt.format(question=question))
        return response.content.strip()
