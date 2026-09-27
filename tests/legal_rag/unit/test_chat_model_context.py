"""The served /v1/chat model must get a context window its prompts fit in.

Measured with Qwen3's tokenizer on 2026-09-27: answer prompts with five evidence
blocks run 5,900-6,300 tokens, but Ollama loaded qwen3:4b at n_ctx 4096 because
build_chat_model passed no num_ctx; the evidence was cut without a warning.
"""
from legal_rag.config import Settings, load_settings
from legal_rag.factory import build_chat_model


def test_configured_context_window_reaches_the_ollama_model():
    model = build_chat_model(Settings(llm={"backend": "ollama", "num_ctx": 8192}))
    assert model.num_ctx == 8192


def test_no_context_window_by_default():
    assert Settings().llm.num_ctx is None
    assert build_chat_model(Settings(llm={"backend": "ollama"})).num_ctx is None


def test_local_profile_fits_answer_prompt_plus_output():
    settings = load_settings(profile="local")
    assert settings.llm.num_ctx is not None
    assert settings.llm.num_ctx >= 6300 + 1024  # measured prompt ceiling + room for the answer
