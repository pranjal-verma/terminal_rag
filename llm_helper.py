import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_llm():
    """
    Select and configure the chat LLM based on environment variables.

    Env vars:
      - LLM_PROVIDER: "GEMINI" (default) or "OPENAI"
      - GEMINI_API_KEY: API key for Gemini (when using GEMINI)
      - OPENAI_API_KEY: API key for OpenAI (when using OPENAI)
    """
    provider = os.getenv("LLM_PROVIDER", "GEMINI").upper()

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

