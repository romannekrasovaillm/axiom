"""E-5: OpenAI-совместимый адаптер §13 (track-2 Stage A, ADR-033).

Зона дельты: ``env/openai_adapter.py`` (порт), ``clients/`` (httpx-транспорт и
CLI калибровки), ``env/tests/``.  Сеть ТОЛЬКО локальные моки
(``httpx.MockTransport`` / фейковый транспорт): внешних вызовов нет,
``llama-server`` не нужен.

Контур C-039: ``env/`` — контур награды RL, HTTP-клиент внутрь него не
импортируется; поэтому тесты разделены — порт проверяется на фейковом
транспорте (без httpx), механизм — на ``clients.openai_http``.

Покрытие: (а) формат запроса; (б) разбор ответа; (в) read-таймаут → понятная
ошибка, один ретрай на connect; (г) ``trust_env=False``; (д) интеграция с
``calibrate`` на мини-наборе; (е) jax/net/httpx не импортируются внутри env/.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
for _p in (str(CASE_DIR), str(CASE_DIR / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clients.openai_http import HttpxTransport, build_openai_factory  # noqa: E402
from env.openai_adapter import (  # noqa: E402
    DEFAULT_BASE_URL,
    OpenAIAdapter,
    OpenAIAdapterError,
    OpenAIReadTimeoutError,
    OpenAIResponseError,
    OpenAIUnavailableError,
    TransportConnectError,
    TransportReadTimeout,
)

CANNED_TEXT = '<tool_call>{"name": "finish", "args": {}}</tool_call>'


# --------------------------------------------------------------------------- #
# Фейки порта (без httpx) и мок-endpoint
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def ok_payload(text: str = CANNED_TEXT, *, with_logprobs: bool = True) -> dict:
    choice: dict = {"index": 0, "message": {"role": "assistant", "content": text}}
    if with_logprobs:
        choice["logprobs"] = {
            "content": [{"token": t, "logprob": -0.5} for t in text.split(" ")]
        }
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [choice],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


class FakeTransport:
    """Порт-транспорт: записывает вызовы, поведение задаётся по номеру вызова."""

    def __init__(self, handler):
        self.calls: list[dict] = []
        self._handler = handler

    def post(self, url: str, *, json: dict, headers: dict):
        self.calls.append({"url": url, "json": json, "headers": headers})
        result = self._handler(len(self.calls))
        if isinstance(result, Exception):
            raise result
        return result


def _fake_adapter(handler=None, **kwargs) -> OpenAIAdapter:
    handler = handler or (lambda n: FakeResponse(200, ok_payload()))
    kwargs.setdefault("transport", FakeTransport(handler))
    return OpenAIAdapter(**kwargs)


# --------------------------------------------------------------------------- #
# (а) формат запроса
# --------------------------------------------------------------------------- #


def test_request_payload_messages_temperature_seed_max_tokens():
    transport = FakeTransport(lambda n: FakeResponse(200, ok_payload()))
    adapter = OpenAIAdapter(
        base_url="http://127.0.0.1:9/v1",
        model="qwen3-4b-instruct",
        seed=3,
        transport=transport,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    adapter.generate(messages, seed=11, max_tokens=64)

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:9/v1/chat/completions"
    body = call["json"]
    assert body["messages"] == messages  # сообщения — как есть
    assert body["temperature"] == 0.0  # детерминизм
    assert body["seed"] == 11
    assert body["max_tokens"] == 64
    assert body["model"] == "qwen3-4b-instruct"
    assert body["stream"] is False


def test_request_seed_defaults_to_adapter_seed():
    transport = FakeTransport(lambda n: FakeResponse(200, ok_payload()))
    OpenAIAdapter(seed=5, transport=transport).generate(
        [{"role": "user", "content": "x"}], max_tokens=8
    )
    assert transport.calls[0]["json"]["seed"] == 5


def test_api_key_added_only_when_present():
    transport = FakeTransport(lambda n: FakeResponse(200, ok_payload()))
    OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "x"}])
    assert "Authorization" not in transport.calls[0]["headers"]
    OpenAIAdapter(api_key="secret-key", transport=transport).generate(
        [{"role": "user", "content": "x"}]
    )
    assert transport.calls[1]["headers"]["Authorization"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# (б) разбор ответа
# --------------------------------------------------------------------------- #


def test_response_parsing_content_and_logprobs():
    result = _fake_adapter().generate([{"role": "user", "content": "go"}])
    assert result.text == CANNED_TEXT
    assert result.token_ids == list(CANNED_TEXT.encode("utf-8"))
    assert result.behavior_logprobs and all(lp == -0.5 for lp in result.behavior_logprobs)
    assert result.policy_version == "openai-local-model"


def test_response_without_logprobs_gives_empty_list():
    adapter = _fake_adapter(lambda n: FakeResponse(200, ok_payload(with_logprobs=False)))
    assert adapter.generate([{"role": "user", "content": "go"}]).behavior_logprobs == []


def test_malformed_response_is_clear_error():
    adapter = _fake_adapter(lambda n: FakeResponse(200, {"choices": []}))
    with pytest.raises(OpenAIResponseError):
        adapter.generate([{"role": "user", "content": "go"}])


def test_http_error_status_is_clear_error():
    adapter = _fake_adapter(lambda n: FakeResponse(500, None, "boom"))
    with pytest.raises(OpenAIResponseError):
        adapter.generate([{"role": "user", "content": "go"}])


def test_missing_transport_is_clear_error():
    adapter = OpenAIAdapter()  # транспорт не задан
    with pytest.raises(OpenAIAdapterError) as excinfo:
        adapter.generate([{"role": "user", "content": "go"}])
    assert "C-039" in str(excinfo.value) or "транспорт" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# (в) таймауты и ретраи (политика порта)
# --------------------------------------------------------------------------- #


def test_read_timeout_raises_clear_error_and_is_not_retried():
    calls = {"n": 0}

    def handler(n):
        calls["n"] = n
        return TransportReadTimeout("read timed out")

    adapter = _fake_adapter(handler)
    with pytest.raises(OpenAIReadTimeoutError) as excinfo:
        adapter.generate([{"role": "user", "content": "go"}])
    assert calls["n"] == 1  # повтор на чтение не делается
    assert "120" in str(excinfo.value) or "p99.9" in str(excinfo.value)


def test_one_retry_on_connect_then_success():
    def handler(n):
        return TransportConnectError("refused") if n == 1 else FakeResponse(200, ok_payload())

    transport = FakeTransport(handler)
    result = OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "go"}])
    assert result.text == CANNED_TEXT
    assert len(transport.calls) == 2  # одна исходная попытка + ровно один ретрай


def test_connect_failure_after_retry_is_clear_error():
    transport = FakeTransport(lambda n: TransportConnectError("refused"))
    with pytest.raises(OpenAIUnavailableError):
        OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "go"}])
    assert len(transport.calls) == 2  # без fallback-эндпоинтов, ровно 1 ретрай


# --------------------------------------------------------------------------- #
# Механизм: httpx-транспорт вне контура награды
# --------------------------------------------------------------------------- #


def test_httpx_transport_request_and_parsing():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_payload())

    transport = HttpxTransport(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(
        base_url="http://127.0.0.1:8080/v1",
        transport=transport,
        seed=4,
    )
    result = adapter.generate([{"role": "user", "content": "go"}], seed=9, max_tokens=32)

    assert result.text == CANNED_TEXT
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["temperature"] == 0.0 and body["seed"] == 9 and body["max_tokens"] == 32
    transport.close()


def test_httpx_transport_maps_connect_error_and_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=ok_payload())

    transport = HttpxTransport(transport=httpx.MockTransport(handler))
    result = OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "go"}])
    assert result.text == CANNED_TEXT
    assert calls["n"] == 2
    transport.close()


def test_httpx_transport_maps_read_timeout_without_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    transport = HttpxTransport(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAIReadTimeoutError):
        OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "go"}])
    assert calls["n"] == 1
    transport.close()


# --------------------------------------------------------------------------- #
# (г) trust_env=False — прокси окружения не наследуются
# --------------------------------------------------------------------------- #


def test_httpx_transport_does_not_trust_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.invalid:1080")
    transport = HttpxTransport()
    try:
        assert transport.client.trust_env is False
        assert not transport.client._mounts  # прокси-маунтов нет
    finally:
        transport.close()


def test_default_base_url_is_local_llama_server():
    assert DEFAULT_BASE_URL == "http://127.0.0.1:8080/v1"


# --------------------------------------------------------------------------- #
# (е) env/ не тянет jax/net/httpx (граница контура награды)
# --------------------------------------------------------------------------- #


def test_env_port_does_not_import_http_clients_or_jax():
    code = (
        "import sys; import env.openai_adapter as m; "
        "assert 'jax' not in sys.modules, sorted(sys.modules); "
        "assert 'net' not in sys.modules, sorted(sys.modules); "
        "assert 'httpx' not in sys.modules, sorted(sys.modules); "
        "assert m.OpenAIAdapter().transport is None; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(CASE_DIR), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------- #
# (д) интеграция с calibrate на мини-наборе (мок-endpoint)
# --------------------------------------------------------------------------- #


def test_calibrate_cli_parses_openai_options():
    from clients.calibrate_openai import build_parser

    args = build_parser().parse_args([
        "--tasks", "/nonexistent/tasks", "--out", "/nonexistent/out",
        "--base-url", "http://127.0.0.1:8080/v1",
        "--served-model", "qwen3-4b-instruct",
        "--model-name", "qwen3",
    ])
    assert args.base_url == "http://127.0.0.1:8080/v1"
    assert args.served_model == "qwen3-4b-instruct"
    assert args.model_name == "qwen3"


def test_calibrate_with_openai_adapter_mini_set(generated, arch_ml, tmp_path):
    """Мини-набор L0 через мок-endpoint: эпизоды идут, отчёт строится."""
    from env import calibrate as calibrate_mod

    src = generated["out"] / "public"
    tasks = tmp_path / "tasks"
    (tasks / "public").mkdir(parents=True)
    (tasks / "holdout").mkdir(parents=True)
    hidden = generated["out"] / "holdout" / "hidden_constraints.yaml"
    if hidden.exists():
        shutil.copy(hidden, tasks / "holdout" / hidden.name)
    for task_id in ("corruption-l0-00", "real-l0-00"):
        shutil.copy(src / f"{task_id}.json", tasks / "public" / f"{task_id}.json")
        shutil.copytree(src / task_id, tasks / "public" / task_id)

    def handler(request: httpx.Request) -> httpx.Response:
        # Мок-сервер всегда отвечает завершающим ходом — эпизод заканчивается.
        return httpx.Response(200, json=ok_payload())

    transport = HttpxTransport(transport=httpx.MockTransport(handler))
    factory = build_openai_factory(seed=11, transport=transport)
    try:
        report = calibrate_mod.calibrate(
            tasks,
            generated["case"],
            tmp_path / "out",
            model_name="openai",
            model_seed=11,
            bin=arch_ml,
            model_factory=factory,
        )
    finally:
        transport.close()

    cells = report["matrix"]["openai"]
    assert len(cells) == 2
    for cell in cells:
        assert cell["termination"] == "finish"
        assert 0.0 <= cell["base_reward"] <= 1.0
    assert 0.0 <= report["aggregate"]["pass_rate"] <= 1.0
    assert (tmp_path / "out" / "calibration-openai.json").exists()
