import json
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

VALIDATOR_PROMPT = """Você é um Perito Jurídico especializado em detecção de falso testemunho e adulteração documental. Sua tarefa é auditar a resposta do sistema comparando-a rigorosamente com o contexto.

Contexto (Verdade Absoluta):
{context}

Afirmação do Sistema:
{statement}

Instruções Cruciais:
1. Se o sistema cita um Artigo que não está no contexto, marque "has_support": false.
2. Se o sistema altera o sentido de um prazo ou penalidade, marque "has_support": false.
3. Se o sistema faz uma interpretação extensiva não escrita no texto, marque "has_support": false.

Responda APENAS em JSON:
{{
    "has_support": boolean,
    "source_fragment": "Citação literal do trecho comprovador",
    "explanation": "Análise técnica do suporte",
    "confidence": float
}}
"""

class HallucinationValidator:
    def __init__(self, model_name: str, base_url: str):
        self.llm = ChatOllama(model=model_name, temperature=0, base_url=base_url, format="json")
        self.prompt = PromptTemplate.from_template(VALIDATOR_PROMPT)

    def validate_statement(self, statement: str, context: str) -> Dict:
        """Valida uma única afirmação contra o contexto."""
        formatted_prompt = self.prompt.format(statement=statement, context=context)
        response = self.llm.invoke(formatted_prompt)
        
        try:
            return json.loads(response.content)
        except Exception as e:
            return {
                "has_support": False, 
                "source_fragment": None, 
                "explanation": f"Erro no processamento: {str(e)}"
            }

    def validate_answer(self, answer: str, context: str) -> List[Dict]:
        """Quebra a resposta em frases e valida cada uma (Simplificado)."""
        # Regex simples para quebrar em frases (considerando terminadores comuns)
        statements = re.split(r'(?<=[.!?])\s+', answer)
        results = []
        for stmt in statements:
            if len(stmt) > 10:  # Ignorar fragmentos muito pequenos
                results.append(self.validate_statement(stmt, context))
        return results

import re # Necessário para o split
