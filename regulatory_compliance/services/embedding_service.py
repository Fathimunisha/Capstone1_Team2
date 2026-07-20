from openai import OpenAI
from regulatory_compliance.core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings using OpenAI embedding model.
    """

    def __init__(self):

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _validate_dimension(
        self,
        embedding: list[float],
    ):
        """
        Validate embedding dimension
        against configured dimension.
        """

        actual_dimension = len(embedding)

        expected_dimension = settings.EMBEDDING_DIMENSION

        if actual_dimension != expected_dimension:

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected: {expected_dimension}, "
                f"Received: {actual_dimension}"
            )

    def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate embedding for a single text.
        Used for user query embedding.
        """

        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )

        embedding = response.data[0].embedding

        # Dimension validation
        self._validate_dimension(embedding)

        return embedding

        logger.info(
            "Generated %s embedding with dimension %s",
            len(embeddings),
            len(embeddings[0]),
        )

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        Used during PDF ingestion.
        """

        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        # Validate every generated embedding
        for embedding in embeddings:

            self._validate_dimension(embedding)

        return embeddings

        logger.info(
            "Generated %s embeddings with dimension %s",
            len(embeddings),
            len(embeddings[0]),
        )
