import json

import httpx
import pytest

from legal_rag.llm import Message, StructuredOutputError
from tests.legal_rag.unit.test_llm import Out


def test_gemini_maps_messages_and_validates_schema():
    from legal_rag.llm.gemini import GeminiChatModel

    def handler(request):
        assert request.url.path.endswith('/models/gemini-3.8-flash:generateContent')
        assert request.headers['x-goog-api-key'] == 'test-key'
        assert 'test-key' not in str(request.url)
        body = json.loads(request.content)
        assert body['systemInstruction']['parts'] == [{'text': 'rules'}]
        assert [c['role'] for c in body['contents']] == ['user', 'model', 'user']
        assert body['generationConfig']['responseJsonSchema'] == Out.model_json_schema()
        assert body['generationConfig']['maxOutputTokens'] == 99
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [
            {'text': 'private reasoning', 'thought': True}, {'text': '{"answer":"ok"}'}]}}]})

    model = GeminiChatModel(model='gemini-3.8-flash', api_key='test-key', transport=httpx.MockTransport(handler))
    result = model.generate([Message(role=r, content=c) for r, c in [
        ('system', 'rules'), ('user', 'q'), ('assistant', 'previous'), ('user', 'next')]],
        response_schema=Out, max_tokens=99)
    assert result.parsed.answer == 'ok'


@pytest.mark.parametrize('body', [
    {'candidates': [{'content': {'parts': [{'text': 'invalid'}]}}]},
    {'promptFeedback': {'blockReason': 'SAFETY'}},
])
def test_gemini_rejects_invalid_or_empty_output(body):
    from legal_rag.llm.gemini import GeminiChatModel
    model = GeminiChatModel(model='m', api_key='key', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body)))
    with pytest.raises(StructuredOutputError):
        model.generate([Message(role='user', content='q')], response_schema=Out)


def test_gemini_requires_key():
    from legal_rag.llm.gemini import GeminiChatModel
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        GeminiChatModel(model='m', api_key='')


def test_gemini_thinking_level_and_combined_token_budget():
    from legal_rag.llm.gemini import GeminiChatModel
    def handler(request):
        config = json.loads(request.content)['generationConfig']
        assert config['thinkingConfig'] == {'thinkingLevel': 'low'}
        assert config['maxOutputTokens'] == 8192
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
    model = GeminiChatModel(model='m', api_key='key', thinking_level='low', transport=httpx.MockTransport(handler))
    assert model.generate([Message(role='user', content='q')], max_tokens=8192).text == 'ok'


def test_gemini_factory_and_yaml_secret_rejection(tmp_path):
    from legal_rag.config import Settings, load_settings
    from legal_rag.factory import build_chat_model
    model = build_chat_model(Settings(llm={'backend': 'gemini', 'model': 'gemini-3.8-flash', 'api_key': 'key'}))
    assert model.name == 'gemini:gemini-3.8-flash'
    (tmp_path / 'default.yaml').write_text('llm:\n  api_key: secret\n')
    with pytest.raises(ValueError, match='llm.api_key'):
        load_settings(profile='default', config_dir=tmp_path)


def test_gemini_plain_text_and_http_errors():
    from legal_rag.llm.gemini import GeminiChatModel
    def handler(request):
        assert 'responseJsonSchema' not in json.loads(request.content)['generationConfig']
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'hello'}]}}]})
    model = GeminiChatModel(model='m', api_key='key', transport=httpx.MockTransport(handler))
    assert model.generate([Message(role='user', content='q')]).text == 'hello'
    model = GeminiChatModel(model='m', api_key='key', transport=httpx.MockTransport(
        lambda request: httpx.Response(429, json={'error': {'message': 'quota exceeded'}})))
    with pytest.raises(httpx.HTTPStatusError):
        model.generate([Message(role='user', content='q')])


def test_cli_reads_gemini_dotenv_and_environment_wins(tmp_path, monkeypatch):
    from legal_rag.cli import settings_from_options
    (tmp_path / '.env').write_text('GEMINI_API_KEY=test-key\nLEGAL_RAG__LLM__BACKEND=gemini\nLEGAL_RAG__LLM__MODEL=gemini-3.8-flash\nLEGAL_RAG__LLM__MAX_TOKENS=8192\nLEGAL_RAG__LLM__GEMINI_THINKING_LEVEL=low\nUNRELATED=value\n')
    settings = settings_from_options(root=tmp_path)
    assert settings.llm.backend == 'gemini'
    assert settings.llm.api_key.get_secret_value() == 'test-key'
    assert settings.llm.max_tokens == 8192
    assert settings.llm.gemini_thinking_level == 'low'
    monkeypatch.setenv('LEGAL_RAG__LLM__MODEL', 'another-model')
    assert settings_from_options(root=tmp_path).llm.model == 'another-model'
    assert settings.hash() == settings.model_copy(update={'llm': settings.llm.model_copy(update={'api_key': None})}).hash()
