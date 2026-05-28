import json
import re
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class CritiqueSchema(BaseModel):
    relevance: float = Field(description="O quanto os chunks recuperados respondem à pergunta (0-1)")
    completeness: float = Field(description="A cobertura dos aspectos jurídicos (0-1)")
    support: float = Field(description="Cada afirmação legal tem referência no documento (0-1)")
    is_approved: bool = Field(description="Se a resposta atinge o threshold de confiança")
    action: str = Field(description="'generate', 'retry' ou 'reject'")

AUDITOR_PROMPT = """Você é um Auditor Jurídico Sênior com 20 anos de experiência em Compliance e Revisão Contratual. Sua missão é garantir que o sistema não forneça informações sem base legal sólida.

Pergunta: {question}

Contexto Recuperado:
{context}

Analise os documentos sob a ótica do rigor normativo (Artigos, Parágrafos, Incisos, Alíneas, Leis, Decretos).
Avalie os seguintes critérios de 0 a 1:
1. RELEVÂNCIA: O texto contém a norma ou cláusula específica solicitada? (Ignore menções genéricas).
2. COMPLETUDE: É possível dar uma resposta conclusiva ou faltam fragmentos cruciais da lei?
3. SUPORTE: Existem referências a diplomas legais (ex: CF/88, CC/02, CLT, CPC)?

Responda APENAS em JSON:
{{
    "relevance": float,
    "completeness": float,
    "support": float,
    "is_approved": boolean,
    "action": "generate" | "retry" | "reject",
    "reasoning": "Breve justificativa técnica"
}}

Threshold: "generate" se relevancia > 0.7 e suporte > 0.6.
"""

class RelevanceAuditor:
    def __init__(self, model_name: str, base_url: str):
        self.llm = ChatOllama(model=model_name, temperature=0, base_url=base_url, format="json")
        self.prompt = PromptTemplate.from_template(AUDITOR_PROMPT)

    def audit(self, question: str, context: str) -> Dict:
        """Avalia o contexto recuperado e retorna um score estruturado."""
        formatted_prompt = self.prompt.format(question=question, context=context)
        response = self.llm.invoke(formatted_prompt)
        
        try:
            critique = json.loads(response.content)
            # Garantir tipos corretos
            critique['relevance'] = float(critique.get('relevance', 0))
            critique['completeness'] = float(critique.get('completeness', 0))
            critique['support'] = float(critique.get('support', 0))
            return critique
        except Exception as e:
            print(f"Erro ao parsear crítica: {e}")
            return {
                "relevance": 0.0,
                "completeness": 0.0,
                "support": 0.0,
                "is_approved": False,
                "action": "retry"
            }
