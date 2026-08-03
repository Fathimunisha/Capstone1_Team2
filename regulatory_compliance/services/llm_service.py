from langchain_openai import ChatOpenAI
from regulatory_compliance.core.config import settings


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            max_tokens=2000,
        )

    def get_llm(self):
        return self.llm
