import os
import logging
from typing import Any, Dict
from fastapi.concurrency import run_in_threadpool
from openai import AzureOpenAI

from ..exceptions import ProviderError
from .base import Provider, EmbeddingProvider
from ...core import settings

logger = logging.getLogger(__name__)


class AzureLLMProvider(Provider):
    def __init__(self,
                 api_key: str | None = None,
                 deployment_name: str = settings.LL_MODEL,
                 opts: Dict[str, Any] = None):
        if opts is None:
            opts = {}

        api_key = api_key or settings.AZURE_OPENAI_API_KEY or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = settings.AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = settings.AZURE_OPENAI_API_VERSION or os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

        if not api_key or not azure_endpoint:
            raise ProviderError("Azure OpenAI API key or endpoint is missing")

        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.deployment = deployment_name
        self.opts = opts
        self.instructions = ""

    def _generate_sync(self, prompt: str, options: Dict[str, Any]) -> str:
        try:
            response = self._client.responses.create(
                model=self.deployment,  # deployment_name
                input=prompt,
                instructions=self.instructions,
                **options,
            )
            return response.output_text
        except Exception as e:
            raise ProviderError(f"Azure OpenAI - error generating response: {e}") from e

    async def __call__(self, prompt: str, **generation_args: Any) -> str:
        if generation_args:
            logger.warning(f"AzureLLMProvider - generation_args not used {generation_args}")
        myopts = {
            "temperature": self.opts.get("temperature", 0),
            "top_p": self.opts.get("top_p", 0.9),
        }
        return await run_in_threadpool(self._generate_sync, prompt, myopts)


class AzureEmbeddingProvider(EmbeddingProvider):
    def __init__(self,
                 api_key: str | None = None,
                 deployment_name: str = settings.EMBEDDING_MODEL):
        api_key = api_key or settings.AZURE_OPENAI_API_KEY or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = settings.AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = settings.AZURE_OPENAI_API_VERSION or os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

        if not api_key or not azure_endpoint:
            raise ProviderError("Azure OpenAI API key or endpoint is missing")

        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self._deployment = deployment_name

    async def embed(self, text: str) -> list[float]:
        try:
            response = await run_in_threadpool(
                self._client.embeddings.create,
                model=self._deployment,  # deployment_name
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise ProviderError(f"Azure OpenAI - error generating embedding: {e}") from e
