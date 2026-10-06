"""E-5: OpenAI-совместимый адаптер §13 (track-2 Stage A, ADR-033).

Зона дельты: ``env/openai_adapter.py`` (новый), ``env/main.py`` (ветка
``--adapter openai``), ``env/tests/``.  Сеть ТОЛЬКО локальные моки
(``httpx.MockTransport``): внешних вызовов нет, ``llama-server`` не нужен.
``jax`` не импортируется (проверяется отдельным процессом).

Покрытие по задаче: (а) формат запроса; (б) разбор ответа; (в) read-таймаут →
понятная ошибка, один ретрай на connect; (г) ``trust_env=False`` (клиент не
видит прокси окружения); (д) интеграция ``calibrate --adapter openai`` на
мини-наборе с мок-сервером; (е) ``jax`` не импортируется.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

CASE_DIR = Path(__file__).resolve().parents[2]
for _p in (str(CASE_DIR), str(CASE_DIR / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env import main as env_main  # noqa: E402
from env.openai_adapter import (  # noqa: E402
    DEFAULT_BASE_URL,
    OpenAIAdapter,
    OpenAIReadTimeoutError,
    OpenAIResponseError,
    OpenAIUnavailableError,
)

CANNED_TEXT = '<tool_call>{"name": "finish", "args": {}}</tool_call>'


# --------------------------------------------------------------------------- #
# Вспомогательное: мок-endpoint
# --------------------------------------------------------------------------- #


def _ok_response(text: str = CANNED_TEXT, *, with_logprobs: bool = True) -> httpx.Response:
    choice: dict = {"index": 0, "message": {"role": "assistant", "content": text}}
    if with_logprobs:
        choice["logprobs"] = {
            "content": [
                {"token": t, "logprob": -0.5}
                for t in text.split(" ")
            ]
        }
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [choice],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    )


def _adapter(**kwargs) -> OpenAIAdapter:
    """Адаптер с мок-транспортом по умолчанию (сеть не трогается)."""
    kwargs.setdefault("transport", httpx.MockTransport(lambda r: _ok_response()))
    return OpenAIAdapter(**kwargs)


# --------------------------------------------------------------------------- #
# (а) формат запроса
# --------------------------------------------------------------------------- #


def test_request_payload_messages_temperature_seed_max_tokens():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok_response()

    adapter = OpenAIAdapter(
        base_url="http://127.0.0.1:9/v1",
        model="qwen3-4b-instruct",
        seed=3,
        transport=httpx.MockTransport(handler),
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    adapter.generate(messages, seed=11, max_tokens=64)

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body["messages"] == messages  # сообщения — как есть
    assert body["temperature"] == 0.0  # детерминизм
    assert body["seed"] == 11
    assert body["max_tokens"] == 64
    assert body["model"] == "qwen3-4b-instruct"
    assert body["stream"] is False


def test_request_seed_defaults_to_adapter_seed():
    seen: list[httpx.Request] = []
    adapter = OpenAIAdapter(
        seed=5,
        transport=httpx.MockTransport(
            lambda r: (seen.append(r), _ok_response())[1]
        ),
    )
    adapter.generate([{"role": "user", "content": "x"}], max_tokens=8)
    assert json.loads(seen[0].content)["seed"] == 5


def test_api_key_added_only_when_present():
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda r: (seen.append(r), _ok_response())[1])
    OpenAIAdapter(transport=transport).generate([{"role": "user", "content": "x"}])
    assert "authorization" not in {k.lower() for k in seen[0].headers}
    seen.clear()
    OpenAIAdapter(api_key="secret-key", transport=transport).generate(
        [{"role": "user", "content": "x"}]
    )
    assert seen[0].headers["authorization"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# (б) разбор ответа
# --------------------------------------------------------------------------- #


def test_response_parsing_content_and_logprobs():
    adapter = _adapter()
    result = adapter.generate([{"role": "user", "content": "go"}])
    assert result.text == CANNED_TEXT
    assert result.token_ids == list(CANNED_TEXT.encode("utf-8"))
    assert result.behavior_logprobs  # собраны из choices[0].logprobs.content
    assert all(lp == -0.5 for lp in result.behavior_logprobs)
    assert result.policy_version == "openai-local-model"


def test_response_without_logprobs_gives_empty_list():
    adapter = _adapter(transport=httpx.MockTransport(lambda r: _ok_response(with_logprobs=False)))
    result = adapter.generate([{"role": "user", "content": "go"}])
    assert result.behavior_logprobs == []


def test_malformed_response_is_clear_error():
    adapter = _adapter(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []})))
    with pytest.raises(OpenAIResponseError):
        adapter.generate([{"role": "user", "content": "go"}])


def test_http_error_status_is_clear_error():
    adapter = _adapter(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    )
    with pytest.raises(OpenAIResponseError):
        adapter.generate([{"role": "user", "content": "go"}])


# --------------------------------------------------------------------------- #
# (в) таймауты и ретраи
# --------------------------------------------------------------------------- #


def test_read_timeout_raises_clear_error_not_traceback():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    adapter = _adapter(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAIReadTimeoutError) as excinfo:
        adapter.generate([{"role": "user", "content": "go"}])
    assert "120" in str(excinfo.value) or "p99.9" in str(excinfo.value)


def test_read_timeout_is_not_retried():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("read timed out", request=request)

    adapter = _adapter(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAIReadTimeoutError):
        adapter.generate([{"role": "user", "content": "go"}])
    assert len(calls) == 1  # повтор на чтение не делается


def test_one_retry_on_connect_then_success():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return _ok_response()

    adapter = _adapter(transport=httpx.MockTransport(handler))
    result = adapter.generate([{"role": "user", "content": "go"}])
    assert result.text == CANNED_TEXT
    assert len(calls) == 2  # одна исходная попытка + ровно один ретрай


def test_connect_failure_after_retry_is_clear_error():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    adapter = _adapter(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAIUnavailableError):
        adapter.generate([{"role": "user", "content": "go"}])
    assert len(calls) == 2  # без fallback-эндпоинтов, ровно 1 ретрай


# --------------------------------------------------------------------------- #
# (г) trust_env=False — прокси окружения не наследуются
# --------------------------------------------------------------------------- #


def test_client_does_not_trust_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.invalid:1080")
    adapter = OpenAIAdapter(base_url=DEFAULT_BASE_URL)
    try:
        assert adapter._client.trust_env is False
        # Ни одного настроенного прокси-маунта нет.
        assert not adapter._client._mounts
    finally:
        adapter.close()


def test_default_base_url_is_local_llama_server():
    assert DEFAULT_BASE_URL == "http://127.0.0.1:8080/v1"


# --------------------------------------------------------------------------- #
# (е) jax не импортируется
# --------------------------------------------------------------------------- #


def test_importing_adapter_does_not_import_jax():
    code = (
        "import sys; import env.openai_adapter as m; "
        "assert 'jax' not in sys.modules, sorted(sys.modules); "
        "assert 'net' not in sys.modules, sorted(sys.modules); "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(CASE_DIR),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------- #
# (д) интеграция с calibrate --adapter openai
# --------------------------------------------------------------------------- #


def test_cli_openai_adapter_choices_and_factory():
    argv = [
        "calibrate", "--tasks", "/nonexistent/tasks", "--out", "/nonexistent/out",
        "--adapter", "openai",
        "--base-url", "http://127.0.0.1:8080/v1",
        "--served-model", "qwen3-4b-instruct",
    ]
    args = env_main.build_parser().parse_args(argv)
    assert args.adapter == "openai"
    factory = env_main._build_model_factory(args)
    assert callable(factory)
    adapter = factory()
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.base_url == "http://127.0.0.1:8080/v1"
    assert adapter.model == "qwen3-4b-instruct"
    adapter.close()


def test_cli_openai_model_name_defaults_to_adapter():
    args = env_main.build_parser().parse_args([
        "calibrate", "--tasks", "/nonexistent/tasks", "--out", "/nonexistent/out",
        "--adapter", "openai",
    ])
    assert args.model_name is None
    assert (args.model_name or args.adapter) == "openai"


def test_calibrate_with_openai_adapter_mini_set(generated, arch_ml, tmp_path):
    """Мини-набор L0 через мок-endpoint: эпизоды идут, отчёт строится."""
    import shutil

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
        return _ok_response()

    def factory():
        return OpenAIAdapter(transport=httpx.MockTransport(handler), seed=11)

    report = calibrate_mod.calibrate(
        tasks,
        generated["case"],
        tmp_path / "out",
        model_name="openai",
        model_seed=11,
        bin=arch_ml,
        model_factory=factory,
    )
    cells = report["matrix"]["openai"]
    assert len(cells) == 2
    for cell in cells:
        assert cell["termination"] == "finish"
        assert 0.0 <= cell["base_reward"] <= 1.0
    assert 0.0 <= report["aggregate"]["pass_rate"] <= 1.0
    assert (tmp_path / "out" / "calibration-openai.json").exists()
