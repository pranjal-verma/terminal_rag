import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()


def _provider(env_var: str, default: str = "GEMINI") -> str:
    return os.getenv(env_var, os.getenv("LLM_PROVIDER", default)).upper()


def get_llm():
    """
    Select and configure the chat LLM based on environment variables.

    Env vars:
      - LLM_PROVIDER: "GEMINI" (default) or "OPENAI"
      - GEMINI_API_KEY: API key for Gemini (when using GEMINI)
      - OPENAI_API_KEY: API key for OpenAI (when using OPENAI)
    """
    provider = _provider("LLM_PROVIDER")

    if provider == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key,
        )

    if provider == "GEMINI":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}. Use 'GEMINI' or 'OPENAI'.")


def get_embeddings():
    """
    Select and configure the embeddings model based on environment variables.

    Env vars:
      - EMBEDDING_PROVIDER: "GEMINI" (default) or "OPENAI" (falls back to LLM_PROVIDER)
      - GEMINI_API_KEY / OPENAI_API_KEY: provider API keys
      - GEMINI_EMBEDDING_MODEL: Gemini embedding model (default: gemini-embedding-2-preview)
      - OPENAI_EMBEDDING_MODEL: OpenAI embedding model (default: text-embedding-3-small)
    """
    provider = _provider("EMBEDDING_PROVIDER")

    if provider == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
        )

    if provider == "GEMINI":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview"),
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {provider!r}. Use 'GEMINI' or 'OPENAI'."
    )


def add_documents_batched(vector_store, documents, batch_size=None, batch_delay=None):
    """
    Add documents to a vector store in batches to stay within Gemini free-tier
    rate limits (100 embed requests/min per model).
    """
    batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    batch_delay = batch_delay or float(os.getenv("EMBEDDING_BATCH_DELAY_SECONDS", "61"))

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)

        remaining = len(documents) - (i + batch_size)
        if remaining > 0:
            print(
                f"  ⏳ Embedded {i + len(batch)}/{len(documents)} chunks, "
                f"waiting {batch_delay:.0f}s..."
            )
            time.sleep(batch_delay)

