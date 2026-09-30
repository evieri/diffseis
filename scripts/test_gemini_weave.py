import os
from pydantic import BaseModel, Field
import weave
from google import genai
from google.genai import types

# Inicializa o rastreamento no Weave (W&B)
weave.init("emmanuel-vieri-ufrn/intro-example")

# Cria cliente oficial do Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class DiffseisQCResult(BaseModel):
    """Schema estruturado para validação e QC geofísico."""
    summary: str = Field(description="Resumo da interpretação ou tarefa geofísica")
    confidence: float = Field(description="Nível de confiança da estimativa de 0 a 1")
    recommendation: str = Field(description="Recomendação de próximos passos")


@weave.op
def ask_gemini(
    prompt: str,
    system_prompt: str = "Você é um assistente sênior em geofísica e inteligência artificial.",
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
) -> str:
    """Envia prompt de texto ao Gemini com rastreamento no Weave."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text or ""


@weave.op
def ask_gemini_structured(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
) -> DiffseisQCResult:
    """Exemplo de saída estrita tipada via Pydantic rastreada no Weave."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DiffseisQCResult,
            temperature=0.1,
        ),
    )
    return DiffseisQCResult.model_validate_json(response.text)


if __name__ == "__main__":
    print("=== 1. Executando chamada de texto simples ===")
    prompt_texto = "Qual é o papel da separação de difrações sísmicas na resolução de falhas geológicas?"
    resultado_texto = ask_gemini(prompt_texto)
    print("Resposta:\n", resultado_texto)

    print("\n=== 2. Executando chamada estruturada (Pydantic) ===")
    prompt_qc = "Analise se um dado sísmico com forte presença de ruído coerente de reflexão requer filtragem antes da difusão."
    resultado_estruturado = ask_gemini_structured(prompt_qc)
    print("Resultado estruturado (Pydantic):", resultado_estruturado.model_dump())
