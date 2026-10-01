from typing import Optional, Any, Union, List
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


def _get_text_content(content: Union[str, List[Any]]) -> str:
    """Safely extracts plain text from LangChain message content (handles str and list blocks)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory function returning a configured LangChain Chat Model across multiple providers.

    Supported providers: 'gemini', 'openai', 'anthropic', 'groq', 'ollama'.
    Defaults to settings configured in `src.config.settings`.
    """
    provider_name = (provider or settings.LLM_PROVIDER).lower()
    model_name = model or settings.LLM_MODEL
    key = api_key or settings.LLM_API_KEY

    if provider_name == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Missing package. Run: `uv add langchain-google-genai`")

        if not key:
            raise ValueError("LLM_API_KEY is required for Gemini provider.")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
            **kwargs,
        )

    elif provider_name == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Missing package. Run: `uv add langchain-openai`")

        if not key:
            raise ValueError("LLM_API_KEY is required for OpenAI provider.")

        return ChatOpenAI(
            model=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs,
        )

    elif provider_name in ("anthropic", "claude"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Missing package. Run: `uv add langchain-anthropic`")

        if not key:
            raise ValueError("LLM_API_KEY is required for Anthropic provider.")

        return ChatAnthropic(
            model_name=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs,
        )

    elif provider_name == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Missing package. Run: `uv add langchain-groq`")

        if not key:
            raise ValueError("LLM_API_KEY is required for Groq provider.")

        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
            **kwargs,
        )

    elif provider_name == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Missing package. Run: `uv add langchain-ollama`")

        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider_name}'. "
            "Supported providers: 'gemini', 'openai', 'anthropic', 'groq', 'ollama'."
        )


class LLMTestResponse(BaseModel):
    """Pydantic model for structured output testing."""

    status: str = Field(description="Operational status, e.g., 'SUCCESS'")
    provider: str = Field(description="Name of the active provider")
    model_name: str = Field(description="Name of the active model")
    summary: str = Field(description="Brief test response summary")


if __name__ == "__main__":
    print(f"\n[LLM Service Verification]")
    print(f"Provider   : '{settings.LLM_PROVIDER}'")
    print(f"Model      : '{settings.LLM_MODEL}'")
    print("=" * 85)

    try:
        # Initialize default LLM instance
        llm = get_llm()

        # 1. Test basic text generation
        print("\n1. Testing standard completion...")
        response = llm.invoke("In one sentence, what is the California Consumer Privacy Act (CCPA)?")
        text_content = _get_text_content(response.content)
        print(f"Response:\n{text_content.strip()}\n")

        # 2. Test structured output capability
        print("2. Testing structured output schema support...")
        structured_llm = llm.with_structured_output(LLMTestResponse)
        test_out = structured_llm.invoke(
            f"Respond with status 'SUCCESS', provider '{settings.LLM_PROVIDER}', "
            f"model_name '{settings.LLM_MODEL}', and summarize CCPA scope in 10 words."
        )
        print(f"Structured Output:\n{test_out.model_dump_json(indent=2)}")

    except Exception as e:
        print(f"\nLLM service test failed: {e}")