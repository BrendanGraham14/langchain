"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self._embed_documents(texts)

    @abstractmethod
    def _embed_query(self, text: str) -> List[float]:
        pass

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_query(text)

    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return await self._aembed_documents(texts)

    async def _aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return await self._aembed_query(text)
