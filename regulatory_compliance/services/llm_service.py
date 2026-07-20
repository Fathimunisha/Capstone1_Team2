import os
import logging
from langchain_openai import ChatOpenAI
from regulatory_compliance.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Wrapper for OpenAI LLM calls.
    """

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            # temperature=0,
            max_tokens=2000,
        )

    def generate_answer(self, question: str, context: str) -> str:

        prompt = f"""
        You are a Regulatory Compliance assistant.
        Answer only from the provided context.
        If information is not available, say:
        "Information not available in provided documents."

        Question:
        {question}

        Context:
        {context}

        Answer:
        """

        response = self.llm.invoke(prompt)

        return response.content
