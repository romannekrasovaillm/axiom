"""T-s1..T-b2 — пайплайн агентных эпизодов axiom-domain-ds-v1 (ADR-020, дельта-1).

Проверяются четыре ступени пайплайна ``tools/axiom_ds``:

* **скраб** (T-s1, T-s3) — секреты не доживают до выхода ни в одном из объявленных
  классов; ключи с hash-именами (sha256/md5/hash/salt/checksum) правилом ``hex_env``
  не трогаются — их значения суть доменные данные (дельта-3);
* **deny-list** (T-s2) — запретные каталоги и файлы не открываются вовсе (не «чистятся»,
  а пропускаются до чтения), счётчик ``sessions_denied`` отделён от ``parse_errors``;
* **эпизодизация** (T-e1) — граница эпизода проходит по пользовательскому запросу,
  tool-цикл остаётся внутри эпизода;
* **верификация** (T-v1..T-v4) — класс исхода механический: контракт сессии или
  парный harness-отчёт, никакой интерпретации прозой (AD-2); ``verified-partial``
  (дельта-3) — ``partial`` плюс зелёный сьют в последних tool-результатах, в SFT
  только с флагом ``verification: partial-green``;
* **дедуп** (T-d1, T-d2) — точные хеши блоков и MinHash near-dup;
* **CLI** (T-b1, T-b2) — ``--limit 0`` даёт пустой валидный jsonl + числовой отчёт,
  полный прогон на синтетике даёт скрабленный jsonl и счётчики классов.

Все фикстуры синтетические. Реальные сессии и реальные секреты в тесты не попадают:
секреты-приманки собраны из очевидно тестовых значений.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from axiom_ds import build as build_mod  # noqa: E402
from axiom_ds import dedup as dedup_mod  # noqa: E402
from axiom_ds import episodes as ep_mod  # noqa: E402
from axiom_ds import scrub as scrub_mod  # noqa: E402
from axiom_ds import verify as verify_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# Секреты-приманки (синтетические, не боевые)
# --------------------------------------------------------------------------- #

OPENAI_KEY = "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_AbCdEf0123456789AbCdEf0123456789AbCd"
SLACK_TOKEN = "xox" + "b-123456789012-abcdefghijklmnop"  # склейка: GitHub push-protection не видит собранного паттерна
BEARER_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.examplepayloadpart"
PRIVATE_KEY_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAxTESTONLYxNOTAREALKEYxMIIEowIBAAKCAQEA\n"
    "-----END RSA PRIVATE KEY-----"
)
URL_WITH_PASSWORD = "postgres://admin:S3cretPw42@db.internal:5432/warehouse"
ENV_ASSIGNMENT = "AXIOM_SERVICE_TOKEN=Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0"
HEX_SECRET_ASSIGNMENT = "AXIOM_SIGNING_SALT=0123456789abcdef0123456789abcdef"
JWT_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJheGlvbS10ZXN0In0.QWERTYuiopASDFghjklZXCVbnm"
)

#: пары «имя правила → фрагмент, который обязан исчезнуть из вычищенного текста».
SECRET_SAMPLES = {
    "openai": OPENAI_KEY,
    "aws": AWS_ACCESS_KEY,
    "github": GITHUB_TOKEN,
    "slack": SLACK_TOKEN,
    "bearer": BEARER_TOKEN.split(" ", 1)[1],
    "private_key": "MIIEowIBAAKCAQEAxTESTONLYxNOTAREALKEYxMIIEowIBAAKCAQEA",
    "url_password": "S3cretPw42",
    "env_assignment": "Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0",
    "hex_env": "0123456789abcdef0123456789abcdef",
    "jwt": JWT_TOKEN,
}

REDACTED = "<REDACTED>"


def poisoned_text() -> str:
    """Текст, в который подмешаны все приманки, в окружении обычной прозы."""
    return "\n".join(
        [
            f"export OPENAI_API_KEY={OPENAI_KEY}",
            f"aws_access_key_id = {AWS_ACCESS_KEY}",
            f"git remote set-url origin https://{GITHUB_TOKEN}@github.com/me/repo",
            f"slack webhook {SLACK_TOKEN}",
            f"Authorization: {BEARER_TOKEN}",
            PRIVATE_KEY_BLOCK,
            f"connect via {URL_WITH_PASSWORD}",
            f"{ENV_ASSIGNMENT}",
            f"{HEX_SECRET_ASSIGNMENT}",
            f"token={JWT_TOKEN}",
            "обычный текст без секретов: правка tools/axiom_ds/build.py",
        ]
    )


# --------------------------------------------------------------------------- #
# Синтетические сессии
# --------------------------------------------------------------------------- #


def ev_user_text(text: str, ts: str, sid: str, uuid: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "sessionId": sid,
        "timestamp": ts,
        "isSidechain": False,
        "message": {"role": "user", "content": text},
    }


def ev_assistant(blocks: list, ts: str, sid: str, uuid: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": None,
        "sessionId": sid,
        "timestamp": ts,
        "isSidechain": False,
        "message": {"role": "assistant", "content": blocks},
    }


def ev_tool_result(tool_use_id: str, content: str, ts: str, sid: str, uuid: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "sessionId": sid,
        "timestamp": ts,
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
            ],
        },
    }


def ev_text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def ev_tool_use(tool_id: str, name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def ev_system(subtype: str, sid: str, ts: str) -> dict:
    return {"type": "system", "subtype": subtype, "sessionId": sid, "timestamp": ts}


def write_jsonl(path: Path, events: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return path


def two_tool_cycle_session(sid: str = "sess-0001", out: str = "готово") -> list:
    """Сессия из двух эпизодов: по два tool-цикла в каждом, финал — текстовый ответ."""
    ts = "2026-09-01T10:{:02d}:00.000Z"
    events = [
        ev_user_text("Задача А: собери отчёт", ts.format(0), sid, "u1"),
        ev_assistant(
            [ev_text_block("Смотрю репозиторий"), ev_tool_use("t1", "Bash", {"command": "ls"})],
            ts.format(1),
            sid,
            "a1",
        ),
        ev_tool_result("t1", "build.py\nepisodes.py", ts.format(2), sid, "r1"),
        ev_assistant(
            [ev_tool_use("t2", "Read", {"file_path": "build.py"})], ts.format(3), sid, "a2"
        ),
        ev_tool_result("t2", "1\timport json", ts.format(4), sid, "r2"),
        ev_assistant([ev_text_block(f"Отчёт собран, {out}")], ts.format(5), sid, "a3"),
        ev_user_text("Задача Б: прогони тесты", ts.format(6), sid, "u2"),
        ev_assistant(
            [ev_tool_use("t3", "Bash", {"command": "pytest -q"})], ts.format(7), sid, "a4"
        ),
        ev_tool_result("t3", "3 passed", ts.format(8), sid, "r3"),
        ev_assistant([ev_text_block("Тесты зелёные")], ts.format(9), sid, "a5"),
    ]
    return events


def contract_text(status: str, **extra) -> str:
    payload = {
        "status": status,
        "assumptions": [],
        "open_questions": [],
        "conflicts_with_prior_decisions": [],
    }
    payload.update(extra)
    return "Готово.\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n"


# --------------------------------------------------------------------------- #
# T-s1 — скраб ловит все объявленные паттерны
# --------------------------------------------------------------------------- #


def test_ts1_scrub_text_removes_every_seeded_secret():
    cleaned = scrub_mod.scrub_text(poisoned_text())
    for name, secret in SECRET_SAMPLES.items():
        assert secret not in cleaned, f"секрет {name} дожил до выхода скраба"
    assert cleaned.count(REDACTED) >= len(SECRET_SAMPLES)
    # обычный текст не пострадал
    assert "правка tools/axiom_ds/build.py" in cleaned


def test_ts1_scrub_reports_redaction_counts_per_pattern():
    stats = scrub_mod.ScrubStats()
    scrub_mod.scrub_text_stats(poisoned_text(), stats)
    assert stats.total >= len(SECRET_SAMPLES)
    assert set(stats.by_pattern) >= {
        "openai",
        "aws",
        "github",
        "slack",
        "bearer",
        "private_key",
        "url_password",
        "env_assignment",
        "jwt",
    }


def test_ts1_scrub_text_is_idempotent():
    once = scrub_mod.scrub_text(poisoned_text())
    assert scrub_mod.scrub_text(once) == once


def test_ts1_complete_private_key_block_leaves_no_markers():
    cleaned = scrub_mod.scrub_text(PRIVATE_KEY_BLOCK)
    assert "PRIVATE KEY" not in cleaned
    assert cleaned.strip() == REDACTED


def test_ts1_dangling_private_key_header_is_redacted():
    """Обрезанное чтение .pem: заголовок и тело есть, футера нет.

    Так выглядит вывод чтения с --limit и результат поиска по файлу: правило
    «целого блока» такое не ловит, а ключевой материал утекает.
    """
    body = "MIIEvwIBADAQABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghij"
    text = f'let pem = "-----BEGIN PRIVATE KEY-----\n{body}\n'
    cleaned = scrub_mod.scrub_text(text)
    assert body not in cleaned
    assert "PRIVATE KEY" not in cleaned


def test_ts1_orphan_private_key_markers_are_redacted():
    """Осиротевшие маркеры ключа не должны переживать скраб."""
    for marker in (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ):
        assert "PRIVATE KEY" not in scrub_mod.scrub_text(f"see {marker} here")


def test_ts1_scrub_session_walks_nested_blocks():
    session = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                ev_text_block(f"ключ {OPENAI_KEY}"),
                ev_tool_use("t1", "Bash", {"command": f"curl -H 'Authorization: {BEARER_TOKEN}'"}),
                {"type": "tool_result", "tool_use_id": "t1", "content": URL_WITH_PASSWORD},
            ],
        },
        "cwd": "/home/roman/axiom",
    }
    stats = scrub_mod.ScrubStats()
    cleaned = scrub_mod.scrub_session(session, stats)
    dumped = json.dumps(cleaned, ensure_ascii=False)
    for secret in (OPENAI_KEY, BEARER_TOKEN.split(" ", 1)[1], "S3cretPw42"):
        assert secret not in dumped
    assert stats.total >= 3
    # исходный объект не мутирован
    assert OPENAI_KEY in json.dumps(session, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# T-s2 — deny-list: запретные пути не читаются вовсе
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "relpath",
    ["API-keys/tokens.jsonl", ".ssh/id_rsa", ".aws/credentials", "certs/server.pem", "x.key"],
)
def test_ts2_is_denied_path_flags_forbidden_paths(tmp_path, relpath):
    assert scrub_mod.is_denied_path(tmp_path / relpath) is True


@pytest.mark.parametrize(
    "relpath",
    ["project/session.jsonl", "project/notes.md", "keys_session.jsonl", "api-key-usage.jsonl"],
)
def test_ts2_is_denied_path_allows_ordinary_session_paths(tmp_path, relpath):
    assert scrub_mod.is_denied_path(tmp_path / relpath) is False


def test_ts2_denied_files_are_not_read_and_not_parsed(tmp_path):
    source = tmp_path / "projects"
    canary = "CANARY-LEAK-" + OPENAI_KEY
    # файл в запретном каталоге — заведомо невалидный jsonl: если бы его читали,
    # он попал бы в parse_errors, а не в sessions_denied.
    (source / "API-keys").mkdir(parents=True)
    (source / "API-keys" / "tokens.jsonl").write_text(
        "NOT JSON {{{ " + canary, encoding="utf-8"
    )
    (source / ".ssh").mkdir(parents=True)
    (source / ".ssh" / "session.jsonl").write_text("NOT JSON {{{ " + canary, encoding="utf-8")
    (source / "certs").mkdir(parents=True)
    (source / "certs" / "session.pem").write_text("NOT JSON {{{ " + canary, encoding="utf-8")
    write_jsonl(source / "ok" / "real.jsonl", two_tool_cycle_session(sid="sess-keep"))

    out = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"
    report = build_mod.run_build(source=[source], out=out, report=report_path)

    assert report["sessions_denied"] == 3
    assert report["sessions_processed"] == 1
    assert report["parse_errors"] == 0
    assert canary not in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# T-e1 — эпизодизация режет по tool-циклам
# --------------------------------------------------------------------------- #


def test_te1_episode_boundaries_follow_user_requests(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", two_tool_cycle_session())
    eps = ep_mod.load_episodes(path)

    assert len(eps) == 2

    first, second = eps
    assert first.source_session == "sess-0001"
    assert [(t.role, t.kind) for t in first.turns] == [
        ("user", "text"),
        ("assistant", "text"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "text"),
    ]
    assert first.turns[0].content == "Задача А: собери отчёт"
    # ход tool_call несёт и имя инструмента, и его аргументы (иначе агентная
    # траектория теряет само действие)
    call = json.loads(first.turns[2].content)
    assert call["name"] == "Bash"
    assert call["input"] == {"command": "ls"}
    assert first.turns[3].content == "build.py\nepisodes.py"
    assert first.turns[-1].content == "Отчёт собран, готово"
    assert first.started_at == "2026-09-01T10:00:00.000Z"
    assert first.ended_at == "2026-09-01T10:05:00.000Z"

    assert second.turns[0].content == "Задача Б: прогони тесты"
    assert [json.loads(t.content)["input"]["command"] for t in second.turns if t.kind == "tool_call"] == [
        "pytest -q"
    ]
    assert second.started_at == "2026-09-01T10:06:00.000Z"


def test_te1_tool_call_input_is_preserved_and_tool_result_is_scrubbed(tmp_path):
    sid = "sess-0002"
    events = [
        ev_user_text("утечка?", "2026-09-01T11:00:00.000Z", sid, "u1"),
        ev_assistant(
            [ev_tool_use("t1", "Bash", {"command": f"echo {OPENAI_KEY}"})],
            "2026-09-01T11:00:01.000Z",
            sid,
            "a1",
        ),
        ev_tool_result("t1", f"stdout {GITHUB_TOKEN}", "2026-09-01T11:00:02.000Z", sid, "r1"),
        ev_assistant([ev_text_block("проверил")], "2026-09-01T11:00:03.000Z", sid, "a2"),
    ]
    path = write_jsonl(tmp_path / "s.jsonl", events)
    eps = ep_mod.load_episodes(path)
    dumped = json.dumps([e.to_dict() for e in eps], ensure_ascii=False)
    assert OPENAI_KEY not in dumped
    assert GITHUB_TOKEN not in dumped
    assert REDACTED in dumped


def test_te1_episode_ids_are_deterministic_and_unique(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", two_tool_cycle_session(sid="sess-0003"))
    first = [e.id for e in ep_mod.load_episodes(path)]
    second = [e.id for e in ep_mod.load_episodes(path)]
    assert first == second
    assert len(set(first)) == len(first) == 2


def test_te1_tool_only_user_events_do_not_open_new_episode(tmp_path):
    sid = "sess-0004"
    events = [
        ev_system("local_command", sid, "2026-09-01T12:00:00.000Z"),
        ev_user_text("запрос", "2026-09-01T12:00:01.000Z", sid, "u1"),
        ev_assistant([ev_tool_use("t1", "Bash", {"command": "ls"})], "2026-09-01T12:00:02.000Z", sid, "a1"),
        ev_tool_result("t1", "ok", "2026-09-01T12:00:03.000Z", sid, "r1"),
        ev_assistant([ev_text_block("ответ")], "2026-09-01T12:00:04.000Z", sid, "a2"),
    ]
    path = write_jsonl(tmp_path / "s.jsonl", events)
    eps = ep_mod.load_episodes(path)
    assert len(eps) == 1
    assert eps[0].started_at == "2026-09-01T12:00:01.000Z"
    assert eps[0].ended_at == "2026-09-01T12:00:04.000Z"


# --------------------------------------------------------------------------- #
# T-v1..T-v3 — класс верификации
# --------------------------------------------------------------------------- #


def episode_with_final(text: str, sid: str = "sess-v") -> ep_mod.Episode:
    events = [
        ev_user_text("сделай", "2026-09-02T10:00:00.000Z", sid, "u1"),
        ev_assistant([ev_text_block(text)], "2026-09-02T10:00:01.000Z", sid, "a1"),
    ]
    return ep_mod.split_episodes(events, sid)[0]


def test_tv1_complete_contract_is_verified_complete():
    ep = episode_with_final(contract_text("complete"))
    assert verify_mod.classify_episode(ep) == verify_mod.VERIFIED_COMPLETE


def test_tv1_harness_report_complete_is_verified_complete():
    ep = episode_with_final("Готово, отчёт без контракта")
    assert verify_mod.classify_episode(ep, harness_status="complete") == verify_mod.VERIFIED_COMPLETE


def test_tv2_blocked_contract_is_verified_failed():
    ep = episode_with_final(contract_text("blocked"))
    assert verify_mod.classify_episode(ep) == verify_mod.VERIFIED_FAILED


def test_tv2_conflicts_and_failed_statuses_are_verified_failed():
    for status in ("conflicts", "failed"):
        ep = episode_with_final(contract_text(status))
        assert verify_mod.classify_episode(ep) == verify_mod.VERIFIED_FAILED, status
    ep = episode_with_final(contract_text("complete"))
    assert (
        verify_mod.classify_episode(ep, harness_status="blocked") == verify_mod.VERIFIED_FAILED
    )


def test_tv3_partial_and_absent_contract_are_unverified():
    assert verify_mod.classify_episode(episode_with_final(contract_text("partial"))) == (
        verify_mod.UNVERIFIED
    )
    assert verify_mod.classify_episode(episode_with_final("просто ответ")) == verify_mod.UNVERIFIED


def test_tv3_malformed_contract_is_unverified():
    assert (
        verify_mod.classify_episode(episode_with_final("```json\n{status: complete}\n```"))
        == verify_mod.UNVERIFIED
    )


def test_tv3_contract_must_be_trailing_not_quoted_mid_episode():
    # контракт в середине эпизода перекрыт более поздним ответом без статуса
    ep = episode_with_final("```json\n{\"status\": \"complete\"}\n```\n\nещё поработаю")
    assert verify_mod.classify_episode(ep) == verify_mod.UNVERIFIED


# --------------------------------------------------------------------------- #
# T-v4 — verified-partial: partial + механический признак зелёного сьюта
# --------------------------------------------------------------------------- #

GREEN_SUITE_RESULT = "============================= 12 passed in 3.41s =============================="
RED_SUITE_RESULT = "FAILED tests/test_x.py::test_y - AssertionError: 1 != 2"


def episode_with_tool_tail(
    results: list[str], final: str | None = None, sid: str = "sess-green"
) -> ep_mod.Episode:
    """Эпизод с хвостом tool-результатов: материалы для признака зелёного сьюта."""
    stamp = "2026-09-04T10:{:02d}:00.000Z"

    def at(minute: int) -> str:
        return stamp.format(minute)

    events = [ev_user_text("прогони тесты", at(0), sid, "u1")]
    for index, result in enumerate(results):
        events.append(
            ev_assistant(
                [ev_tool_use(f"t{index}", "Bash", {"command": "pytest -q"})],
                at(2 * index + 1),
                sid,
                f"a{index}",
            )
        )
        events.append(ev_tool_result(f"t{index}", result, at(2 * index + 2), sid, f"r{index}"))
    if final is not None:
        events.append(ev_assistant([ev_text_block(final)], at(2 * len(results) + 1), sid, "af"))
    return ep_mod.split_episodes(events, sid)[0]


def test_tv4_partial_with_green_suite_is_verified_partial():
    """partial + зелёный сьют в tool-результатах → SFT-допуск с флагом."""
    ep = episode_with_tool_tail([GREEN_SUITE_RESULT], final=contract_text("partial"))
    assert verify_mod.apply_class(ep) == verify_mod.VERIFIED_PARTIAL
    assert ep.evidence == verify_mod.EVIDENCE_PARTIAL_GREEN
    row = ep.to_dict()
    assert row["class"] == verify_mod.VERIFIED_PARTIAL
    assert row["verification"] == "partial-green"
    assert verify_mod.VERIFIED_PARTIAL in verify_mod.SFT_CLASSES


def test_tv4_partial_without_green_suite_stays_unverified():
    """partial без механического подтверждения — прежний unverified (без флага)."""
    ep = episode_with_tool_tail(["компиляция прошла, отчёта сьюта нет"],
                                final=contract_text("partial"))
    assert verify_mod.apply_class(ep) == verify_mod.UNVERIFIED
    assert ep.evidence == verify_mod.EVIDENCE_CONTRACT
    assert ep.to_dict()["verification"] is None


def test_tv4_green_suite_without_partial_status_does_not_change_class():
    """Зелёный сьют — снисхождение ровно на шаг: complete остаётся complete."""
    complete = episode_with_tool_tail([GREEN_SUITE_RESULT], final=contract_text("complete"))
    assert verify_mod.classify_episode(complete) == verify_mod.VERIFIED_COMPLETE
    # без контракта зелёный сьют класса не поднимает — статуса partial нет
    no_contract = episode_with_tool_tail([GREEN_SUITE_RESULT])
    assert verify_mod.classify_episode(no_contract) == verify_mod.UNVERIFIED


def test_tv4_harness_partial_with_green_suite_is_verified_partial():
    """Источник `partial` — и контракт сессии, и парный harness-отчёт."""
    ep = episode_with_tool_tail([GREEN_SUITE_RESULT])
    assert verify_mod.classify_episode(ep, harness_status="partial") == (
        verify_mod.VERIFIED_PARTIAL
    )
    assert verify_mod.classify_episode(
        episode_with_tool_tail(["ничего про сьют"]), harness_status="partial"
    ) == verify_mod.UNVERIFIED


def test_tv4_red_line_in_same_results_cancels_green():
    """«FAILED»/«ERROR» в тех же результатах отменяет признак зелёного сьюта."""
    ep = episode_with_tool_tail([GREEN_SUITE_RESULT + "\n" + RED_SUITE_RESULT],
                                final=contract_text("partial"))
    assert verify_mod.classify_episode(ep) == verify_mod.UNVERIFIED


def test_tv4_window_is_last_five_tool_results():
    """Признак ищется по последним 5 tool-результатам, а не по всему эпизоду."""
    outside = episode_with_tool_tail([GREEN_SUITE_RESULT] + ["шум без отчёта"] * 5,
                                     final=contract_text("partial"))
    assert verify_mod.classify_episode(outside) == verify_mod.UNVERIFIED  # сьют за окном
    inside = episode_with_tool_tail(["шум без отчёта"] * 5 + [GREEN_SUITE_RESULT],
                                    final=contract_text("partial"))
    assert verify_mod.classify_episode(inside) == verify_mod.VERIFIED_PARTIAL
    assert verify_mod.GREEN_SUITE_WINDOW == 5


def test_tv4_assistant_text_is_not_a_green_suite():
    """Проза ассистента («тесты зелёные») признак не даёт: смотрится вывод среды."""
    stamp = "2026-09-04T11:{:02d}:00.000Z"
    events = [
        ev_user_text("прогони тесты", stamp.format(0), "sess-text", "u1"),
        ev_assistant([ev_text_block("тесты passed, всё зелёное")], stamp.format(1),
                     "sess-text", "a1"),
        ev_tool_result("t0", "лог без отчёта сьюта", stamp.format(2), "sess-text", "r0"),
        ev_assistant([ev_text_block(contract_text("partial"))], stamp.format(3),
                     "sess-text", "a2"),
    ]
    ep = ep_mod.split_episodes(events, "sess-text")[0]
    assert verify_mod.classify_episode(ep) == verify_mod.UNVERIFIED


def test_tv4_partial_green_episode_lands_in_sft_component(tmp_path):
    """SFT-компонент: verified-partial пишется с флагом, unverified — не пишется."""
    source = tmp_path / "projects"
    stamp = "2026-09-05T10:{:02d}:00.000Z"

    partial_green = unique_tool_cycle_session("sess-partial-green", "delta", contract="partial")
    partial_green[6] = ev_tool_result(
        "t2", f"delta: {GREEN_SUITE_RESULT}", stamp.format(6), "sess-partial-green", "r2"
    )
    write_jsonl(source / "p1" / "partial-green.jsonl", partial_green)

    partial_dry = unique_tool_cycle_session("sess-partial-dry", "epsilon", contract="partial")
    write_jsonl(source / "p2" / "partial-dry.jsonl", partial_dry)

    complete = unique_tool_cycle_session("sess-complete", "zeta", contract="complete")
    write_jsonl(source / "p3" / "complete.jsonl", complete)

    out = tmp_path / "episodes-v1.jsonl"
    report_path = tmp_path / "report.json"
    sft = tmp_path / "episodes-v1-sft.jsonl"
    assert build_mod.main([
        "--source", str(source), "--out", str(out), "--report", str(report_path),
        "--sft-out", str(sft),
    ]) == 0

    rows = [json.loads(line) for line in sft.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "SFT-компонент пуст"
    assert all(row["class"] in verify_mod.SFT_CLASSES for row in rows)
    flagged = [row for row in rows if row["class"] == verify_mod.VERIFIED_PARTIAL]
    assert flagged, "verified-partial не попал в SFT-компонент"
    assert all(row["verification"] == "partial-green" for row in flagged)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["by_class"][verify_mod.VERIFIED_PARTIAL] >= 1
    assert report["sft_partial_ready"] == report["by_class_written"][verify_mod.VERIFIED_PARTIAL]
    # классы дельты-1 сохранены — счётчики не подменены новым классом
    assert report["sft_ready"] == report["by_class_written"][verify_mod.VERIFIED_COMPLETE]
    assert report["episodes_written"] == sum(report["by_class_written"].values())
    # объём компоненты E — той же мерой, что K/D/S (для долей карточки)
    assert report["approx_tokens"] > 0 and report["chars"] > 0
    assert report["out_bytes"] > 0 and len(report["out_sha256"]) == 64


# --------------------------------------------------------------------------- #
# T-s3 — hex_env не трогает ключи с hash-именами
# --------------------------------------------------------------------------- #


HEX_HASH_VALUE = "0123456789abcdef0123456789abcdef"  # 32 hex — значение-хеш, не секрет


@pytest.mark.parametrize(
    "key",
    ["sha256", "doc_sha256", "SHA1", "md5", "MD5_CHECKSUM", "img_hash", "hash", "salt",
     "salt_v2", "body_checksum"],
)
def test_ts3_hash_named_keys_are_left_untouched(key):
    """Ключи с hash-именами — доменные данные: правило hex_env к ним не применяется."""
    text = f"{key}={HEX_HASH_VALUE}"
    stats = scrub_mod.ScrubStats()
    assert scrub_mod.scrub_text_stats(text, stats) == text
    assert stats.total == 0


def test_ts3_boundary_of_applied_rules_is_explicit():
    """Граница правил видна в счётчиках: hash-имя не даёт hex_env-замен.

    Имя, оканчивающееся на ``SALT``, остаётся в списке имён правила
    ``env_assignment`` (credential-контекст дельты-1) — там замена прежняя, а
    именно ``hex_env`` к hash-именам не применяется. Граница зафиксирована
    тестом, а не молчанием: снятие ``SALT`` из имён ``env_assignment`` —
    отдельное решение архитектора.
    """
    stats = scrub_mod.ScrubStats()
    cleaned = scrub_mod.scrub_text_stats(f"AXIOM_SIGNING_SALT={HEX_HASH_VALUE}", stats)
    assert HEX_HASH_VALUE not in cleaned            # прежнее правило имён сработало
    assert "hex_env" not in stats.by_pattern        # новое правило к hash-имени не применено
    assert stats.by_pattern.get("env_assignment") == 1


@pytest.mark.parametrize(
    "key", ["OPENAI_API_KEY", "AXIOM_SERVICE_TOKEN", "SERVICE_SECRET", "db_password"]
)
def test_ts3_credential_named_hex_values_are_still_redacted(key):
    """Для credential-имён правило работает как прежде: hex-значение вычищается."""
    cleaned = scrub_mod.scrub_text(f"{key}={HEX_HASH_VALUE}")
    assert HEX_HASH_VALUE not in cleaned
    assert REDACTED in cleaned


def test_ts3_marker_list_is_the_declared_one():
    """Список подстрок зафиксирован константой (ADR-020, дельта-3)."""
    assert scrub_mod.HASH_KEY_MARKERS == ("sha256", "sha1", "md5", "hash", "salt", "checksum")


# --------------------------------------------------------------------------- #
# T-d1 / T-d2 — дедупликация
# --------------------------------------------------------------------------- #


def make_episode(sid: str, index: int, text: str) -> dict:
    events = [
        ev_user_text("запрос", "2026-09-03T10:00:00.000Z", sid, "u1"),
        ev_assistant([ev_text_block(text)], "2026-09-03T10:00:01.000Z", sid, "a1"),
    ]
    return ep_mod.split_episodes(events, sid, start_index=index)[0].to_dict()


def test_td1_exact_duplicates_are_dropped():
    deduper = dedup_mod.Deduper()
    body = "Одинаковый ответ на одинаковый запрос. " * 40

    assert deduper.add(make_episode("sess-a", 0, body)) is True
    assert deduper.add(make_episode("sess-b", 0, body)) is False
    assert deduper.stats.exact == 1
    assert deduper.stats.near == 0
    assert deduper.stats.kept == 1

    # отличимое содержимое проходит
    assert deduper.add(make_episode("sess-c", 0, "совсем другой текст " * 40)) is True
    assert deduper.stats.kept == 2


def test_td2_near_duplicates_on_shuffled_lines_are_dropped():
    deduper = dedup_mod.Deduper()
    lines = [f"строка {i} с достаточно длинным содержимым для шинглов ABCDEFGHIJKLMNOP" for i in range(30)]
    original = "\n".join(lines)
    shuffled = "\n".join(reversed(lines))

    assert dedup_mod.jaccard_estimate(
        dedup_mod.minhash_signature(original), dedup_mod.minhash_signature(shuffled)
    ) >= dedup_mod.JACCARD_THRESHOLD

    assert deduper.add(make_episode("sess-a", 0, original)) is True
    assert deduper.add(make_episode("sess-b", 0, shuffled)) is False
    assert deduper.stats.near == 1
    assert deduper.stats.kept == 1

    distinct = "\n".join(f"иная тема {i} протокол ZYXWVUTSRQPONMLKJIHGFEDCBA" for i in range(30))
    assert deduper.add(make_episode("sess-c", 0, distinct)) is True


def test_td2_normalization_ignores_case_and_whitespace():
    a = dedup_mod.normalize_text("Привет   МИР\n\n  ok")
    b = dedup_mod.normalize_text("привет мир ok")
    assert a == b


def test_td1_exact_fingerprint_ignores_key_order_of_metadata():
    ep_a = make_episode("sess-a", 0, "тело " * 20)
    ep_b = dict(ep_a)
    ep_b["started_at"] = "2030-01-01T00:00:00.000Z"
    assert dedup_mod.exact_fingerprint(ep_a) == dedup_mod.exact_fingerprint(ep_b)


# --------------------------------------------------------------------------- #
# T-b1 / T-b2 — CLI
# --------------------------------------------------------------------------- #


def test_tb1_limit_zero_writes_empty_but_valid_jsonl_and_report(tmp_path):
    source = tmp_path / "projects"
    write_jsonl(source / "p" / "s.jsonl", two_tool_cycle_session(sid="sess-1"))
    out = tmp_path / "nested" / "episodes.jsonl"
    report_path = tmp_path / "report.json"

    rc = build_mod.main(
        [
            "--source",
            str(source),
            "--out",
            str(out),
            "--report",
            str(report_path),
            "--limit",
            "0",
        ]
    )
    assert rc == 0
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == []

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sessions_processed"] == 0
    assert report["episodes_total"] == 0
    assert report["episodes_written"] == 0
    assert report["dedup"]["kept"] == 0
    assert report["redactions"]["total"] == 0
    assert report["status"] == "ok"


def unique_tool_cycle_session(sid: str, salt: str, contract: str | None = None) -> list:
    """Сессия с уникальным по содержанию телом — чтобы near-dup её не съел.

    Синтетические сессии «под копирку» законно схлопываются дедупом, поэтому для
    проверки классов нужны заведомо различные траектории.
    """
    body = " ".join(f"{salt}{i:03d}" for i in range(60))
    ts = "2026-09-01T10:{:02d}:00.000Z"
    events = [
        ev_user_text(f"Задача {sid}: разбери {body}", ts.format(0), sid, "u1"),
        ev_assistant(
            [
                ev_text_block(f"Смотрю {salt}"),
                ev_tool_use("t1", "Bash", {"command": f"grep -rn {salt} {body[:40]}"}),
            ],
            ts.format(1),
            sid,
            "a1",
        ),
        ev_tool_result("t1", f"{salt}: найдено совпадений — {body[40:]}", ts.format(2), sid, "r1"),
        ev_assistant([ev_text_block(f"Отчёт по {salt}: {body[80:]}")], ts.format(3), sid, "a2"),
        ev_user_text(f"Вторая задача {sid}: проверь {body[::-1]}", ts.format(4), sid, "u2"),
        ev_assistant(
            [ev_tool_use("t2", "Bash", {"command": f"pytest -q -k {salt}"})],
            ts.format(5),
            sid,
            "a3",
        ),
        ev_tool_result("t2", f"{salt}: проверки пройдены {body[:60]}", ts.format(6), sid, "r2"),
    ]
    if contract is not None:
        events.append(
            ev_assistant([ev_text_block(contract_text(contract))], ts.format(7), sid, "a4")
        )
    else:
        events.append(
            ev_assistant([ev_text_block(f"Итог {salt}: {body[20:100]}")], ts.format(7), sid, "a4")
        )
    return events


def test_tb2_end_to_end_build_scrubs_and_classifies(tmp_path):
    source = tmp_path / "projects"

    ok_events = unique_tool_cycle_session("sess-ok", "alpha", contract="complete")
    write_jsonl(source / "p1" / "ok.jsonl", ok_events)

    bad_events = unique_tool_cycle_session("sess-bad", "bravo", contract="blocked")
    bad_events[2] = ev_tool_result(
        "t1", f"FAILED with {OPENAI_KEY}", "2026-09-01T10:02:00.000Z", "sess-bad", "r1"
    )
    write_jsonl(source / "p2" / "bad.jsonl", bad_events)

    write_jsonl(source / "p3" / "meh.jsonl", unique_tool_cycle_session("sess-meh", "charlie"))

    # точный дубль первого эпизода sess-ok (события 0..3 — до второго запроса);
    # должен быть снят дедупом
    dup_events = [dict(event, sessionId="sess-dup") for event in ok_events[:4]]
    write_jsonl(source / "p4" / "dup.jsonl", dup_events)

    out = tmp_path / "episodes-v1.jsonl"
    report_path = tmp_path / "report.json"
    rc = build_mod.main(
        ["--source", str(source), "--out", str(out), "--report", str(report_path)]
    )
    assert rc == 0

    raw = out.read_text(encoding="utf-8")
    assert OPENAI_KEY not in raw
    assert "<REDACTED>" in raw

    rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    assert rows, "выход не должен быть пустым"
    for row in rows:
        assert {"id", "source_session", "class", "turns", "started_at", "ended_at"} <= set(row)
        assert row["class"] in {
            verify_mod.VERIFIED_COMPLETE,
            verify_mod.VERIFIED_FAILED,
            verify_mod.UNVERIFIED,
        }

    classes = {row["class"] for row in rows}
    assert verify_mod.VERIFIED_COMPLETE in classes
    assert verify_mod.VERIFIED_FAILED in classes
    assert verify_mod.UNVERIFIED in classes

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sessions_processed"] == 4
    assert report["episodes_total"] >= 5
    assert report["by_class"][verify_mod.VERIFIED_COMPLETE] >= 1
    assert report["by_class"][verify_mod.VERIFIED_FAILED] >= 1
    assert report["dedup"]["exact"] + report["dedup"]["near"] >= 1
    assert report["redactions"]["total"] >= 1
    # by_class — корпус до дедупа, by_class_written — состав датасета;
    # sft_ready — размер SFT-ядра (только verified-complete)
    assert report["episodes_written"] == sum(report["by_class_written"].values())
    assert report["sft_ready"] == report["by_class_written"][verify_mod.VERIFIED_COMPLETE]
    assert report["sft_ready"] >= 1
    assert report["episodes_total"] == sum(report["by_class"].values())
    # отчёт числовой: ни одного значения-секрета в нём
    report_raw = report_path.read_text(encoding="utf-8")
    assert OPENAI_KEY not in report_raw
    assert "<REDACTED>" not in report_raw


def test_tb2_build_is_deterministic(tmp_path):
    source = tmp_path / "projects"
    write_jsonl(source / "p" / "s.jsonl", two_tool_cycle_session(sid="sess-det"))
    outs = []
    for run in ("a", "b"):
        out = tmp_path / f"out-{run}.jsonl"
        build_mod.run_build(source=[source], out=out, report=tmp_path / f"rep-{run}.json")
        outs.append(out.read_text(encoding="utf-8"))
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- #
# Контракт harness-отчётов (источник механической верификации)
# --------------------------------------------------------------------------- #


def test_harness_index_reads_result_json_and_contract_log_line(tmp_path):
    from axiom_ds import harness as harness_mod

    report_dir = tmp_path / "harness"
    result_dir = report_dir / "job-1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps({"status": "complete", "files": ["tools/x.py"]}), encoding="utf-8"
    )
    (report_dir / "hr-20260912-064951-00.log").write_text(
        "Харнесс 'claude-code' завершился: код 0.\n"
        "Контракт результата: status=blocked; assumptions: 1; open_questions: 0\n",
        encoding="utf-8",
    )

    index = harness_mod.HarnessIndex.from_glob(str(report_dir / "*" / "result.json"))
    index.add_glob(str(report_dir / "*.log"))
    statuses = sorted(index.statuses())
    assert statuses == ["blocked", "complete"]
    assert index.size == 2


def test_harness_status_matching_by_time_window(tmp_path):
    from axiom_ds import harness as harness_mod

    report_dir = tmp_path / "harness"
    report_dir.mkdir(parents=True)
    (report_dir / "hr-20260901-100000-00.log").write_text(
        "Контракт результата: status=complete; assumptions: 0; open_questions: 0\n",
        encoding="utf-8",
    )
    index = harness_mod.HarnessIndex.from_glob(str(report_dir / "*.log"))
    assert index.match(started_at="2026-09-01T10:05:00.000Z", ended_at="2026-09-01T10:40:00.000Z") == (
        "complete"
    )
    assert index.match(started_at="2026-09-05T10:05:00.000Z", ended_at="2026-09-05T10:40:00.000Z") is None


# --------------------------------------------------------------------------- #
# T-m1/T-m2, T-c1..T-c3 — сэмплирование шинглов и контейнмент D→K
# (ADR-020, дельта-2: находки боевого прогона — «MinHash-обрушение на длинных
#  документах» и «D↔K — контеймент, а не дубль»)
# --------------------------------------------------------------------------- #


def make_long_pair(target_chars: int, share: float, seed: int = 11) -> tuple[str, str]:
    """Пара документов: доля ``share`` слов общая, остальное — своё.

    Документ A — общий префикс + «свои» слова, документ B — тот же префикс +
    другие «свои» слова. Истинный Jaccard шинглов пары измеряется тестом, а не
    постулируется: фикстура обязана быть той, за которую себя выдаёт.
    """
    import random

    rng = random.Random(seed)
    words = []
    total = 0
    while total < target_chars:
        word = "".join(rng.choice("абвгдежзиклмнопрстуф") for _ in range(rng.randint(4, 11)))
        words.append(word)
        total += len(word) + 1
    cut = max(1, int(len(words) * share))
    left = " ".join(words)
    right = " ".join(words[:cut] + ["z" + word for word in words[cut:]])
    return left, right


def true_jaccard(left: str, right: str) -> float:
    """Истинный Jaccard по 5-граммам: полный перебор шинглов, без выборки."""
    a = set(dedup_mod.shingle_hashes(left, max_shingles=10**9))
    b = set(dedup_mod.shingle_hashes(right, max_shingles=10**9))
    return len(a & b) / len(a | b)


def test_tm1_long_documents_estimate_tracks_true_jaccard():
    """T-m1: 20k символов, истинный Jaccard ≈0.91 → оценка ≥0.8 (регресс обрушения).

    До фикса выборка шинглов брала равномерные РАНГИ отсортированных хешей:
    значения у двух документов расходились, и оценка падала до 0.02–0.17.
    Консистентная выборка (bottom-k по значению хеша) сохраняет общие шинглы
    в обеих выборках, поэтому оценка следует за истинным Jaccard.
    """
    left, right = make_long_pair(20_000, 0.92)
    assert len(left) >= 20_000
    true_value = true_jaccard(left, right)
    assert 0.9 <= true_value <= 0.93, f"фикстура не та: истинный Jaccard {true_value:.3f}"

    left_shingles = dedup_mod.shingle_hashes(left)
    assert len(left_shingles) == dedup_mod.MAX_SHINGLES, "выборка обязана сработать"

    estimate = dedup_mod.jaccard_estimate(
        dedup_mod.minhash_signature(left), dedup_mod.minhash_signature(right)
    )
    assert estimate >= 0.8, f"оценка {estimate:.3f} при истинном Jaccard {true_value:.3f}"
    assert abs(estimate - true_value) < 0.15


def test_tm1_sampling_is_consistent_not_rank_based():
    """Выборка шинглов — консистентная: Jaccard ВЫБОРОК следует за истинным.

    Ранговая выборка брала у двух документов разные значения (квантили сортировки
    расходятся), и Jaccard выборок обрушивался до ≈0.02 — здесь он обязан
    следовать за истинным Jaccard шинглов. Проверка дешевле полной подписи и
    бьёт ровно в свойство выборки: общий шингл входит в выборку обоих или ни одного.
    """
    left, right = make_long_pair(20_000, 0.92)
    left_sample = set(dedup_mod.shingle_hashes(left))
    right_sample = set(dedup_mod.shingle_hashes(right))
    assert len(left_sample) == dedup_mod.MAX_SHINGLES, "выборка обязана сработать"

    sample_jaccard = len(left_sample & right_sample) / len(left_sample | right_sample)
    assert sample_jaccard >= 0.8, f"выборки разошлись: Jaccard выборок {sample_jaccard:.3f}"
    assert abs(sample_jaccard - true_jaccard(left, right)) < 0.15


def test_tm2_short_duplicates_are_still_detected():
    """T-m2: регресс — короткие дубли (выборка не включается) ловятся как раньше."""
    left, right = make_long_pair(500, 0.95, seed=3)
    assert len(dedup_mod.shingle_hashes(left)) < dedup_mod.MAX_SHINGLES
    true_value = true_jaccard(left, right)
    assert true_value >= dedup_mod.JACCARD_THRESHOLD

    estimate = dedup_mod.jaccard_estimate(
        dedup_mod.minhash_signature(left), dedup_mod.minhash_signature(right)
    )
    assert estimate >= dedup_mod.JACCARD_THRESHOLD
    assert abs(estimate - true_value) < 0.15

    deduper = dedup_mod.Deduper()
    assert deduper.add(make_episode("sess-a", 0, left)) is True
    assert deduper.add(make_episode("sess-b", 0, right)) is False
    assert deduper.stats.near == 1


# --- контейнмент ------------------------------------------------------------ #


def filler(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def card_with_chunks(chunks: int, prefix: str = "карточка") -> str:
    """Текст ровно в ``chunks`` окон: длина вскрыта из шага нарезки.

    Фикстура обязана быть той, за которую себя выдаёт: длину задаём по сетке
    окон (``CHUNK_CHARS`` + шаг × (chunks - 1)), а не числом слов.
    """
    step = dedup_mod.CHUNK_CHARS - dedup_mod.CHUNK_OVERLAP_CHARS
    target = dedup_mod.CHUNK_CHARS + step * (chunks - 1)
    text = filler(prefix, target)
    assert len(text) >= target
    return text[:target]


def containment_records(card: str, distillate: str) -> list[tuple[str, str, str]]:
    return [("k-1", "K", card), ("d-1", "D", distillate)]


def inside_distillate(card: str, padding: int = 60) -> str:
    return filler("дистиллят", padding) + " " + card + " " + filler("хвост", 40)


def test_tc1_contained_card_is_dropped_and_container_kept():
    """T-c1: карточка (3 чанка) целиком внутри дистиллята → K снят, D остался."""
    card = card_with_chunks(3)
    assert len(dedup_mod.chunk_windows(card)) == 3, "фикстура: карточка ровно в 3 чанка"
    distillate = inside_distillate(card)

    kept, stats = dedup_mod.containment_dedup(containment_records(card, distillate))

    assert [record[0] for record in kept] == ["d-1"]
    assert stats.dropped == 1
    assert stats.dropped_by_component == {"K": 1}
    assert stats.checked_pairs == 1
    assert stats.coverage_histogram["1.0"] == 1


def test_tc1_containment_is_order_independent():
    """Решения по K не зависят от порядка записей: контейнеры индексируются все."""
    card = card_with_chunks(3)
    records = containment_records(card, inside_distillate(card))

    kept, stats = dedup_mod.containment_dedup(list(reversed(records)))

    assert [record[0] for record in kept] == ["d-1"]
    assert stats.dropped == 1


def test_tc2_partially_overlapping_card_is_kept():
    """T-c2: у карточки с D общий только 40% содержания → карточка остаётся."""
    card_parts = [filler(f"карточка{index}", 20) for index in range(5)]
    card = " ".join(card_parts)
    assert len(dedup_mod.chunk_windows(card)) >= 5
    # в дистиллят попадают только первые две части (40% содержания карточки)
    distillate = filler("дистиллят", 60) + " " + " ".join(card_parts[:2])

    kept, stats = dedup_mod.containment_dedup(containment_records(card, distillate))

    assert {record[0] for record in kept} == {"k-1", "d-1"}
    assert stats.dropped == 0
    assert stats.checked_pairs == 1
    coverages = [float(label) for label in stats.coverage_histogram]
    assert coverages and max(coverages) < 0.8, f"покрытие {coverages} не может доходить до порога"


def test_tc3_threshold_cuts_by_reported_counters():
    """T-c3: срез делает ПОРОГ, и это видно по счётчикам отчёта.

    Фикстура: карточка в 22 чанка, в дистиллят попал её хвост начиная с 600-го
    символа — покрытие 19/22 = 0.864 (замер): 19 окон запроса лежат в дистилляте
    целиком. При пороге 0.8 запись снимается, при 0.9 остаётся — различие даёт
    порог, а не случайность, и корзина покрытия отчёта («не ниже 0.8») это
    подтверждает.
    """
    card = card_with_chunks(22)
    total = len(dedup_mod.chunk_windows(card))
    assert total == 22, "фикстура: карточка ровно в 22 чанка"
    distillate = filler("дистиллят", 60) + " " + card[600:]
    records = containment_records(card, distillate)

    kept_strict, strict = dedup_mod.containment_dedup(records, threshold=0.8)
    kept_lenient, lenient = dedup_mod.containment_dedup(records, threshold=0.9)

    assert [record[0] for record in kept_strict] == ["d-1"], strict.to_dict()
    assert {record[0] for record in kept_lenient} == {"k-1", "d-1"}
    assert strict.dropped == 1 and strict.threshold == 0.8
    assert lenient.dropped == 0 and lenient.threshold == 0.9
    assert strict.checked == 1 and lenient.checked == 1
    # при пороге 0.9 первый же кандидат не успевает добрать порог (даже все
    # оставшиеся чанки его не спасают) — пар не заведено вовсе, и это видно
    assert strict.checked_pairs == 1 and lenient.checked_pairs == 0
    # корзина — нижняя граница покрытия: снято при 0.8, оставлено при 0.9
    assert strict.coverage_histogram["0.8"] == 1, strict.to_dict()
    assert "0.9" not in strict.coverage_histogram


def test_tc3_containment_is_component_agnostic():
    """Ступень не знает про K/D: роли задаются парами (контейнер, содержимое)."""
    inner = filler("блок", 80)
    outer = filler("обёртка", 60) + " " + inner + " " + filler("хвост", 40)
    records = [("y-1", "Y", inner), ("x-1", "X", outer)]

    kept, stats = dedup_mod.containment_dedup(
        records, sides=(("X", "Y"),), threshold=0.8
    )

    assert [record[0] for record in kept] == ["x-1"]
    assert stats.dropped_by_component == {"Y": 1}


def test_tc3_extract_blocks_limits_what_is_compared():
    """``extract_blocks`` задаёт блоки записи: вне блоков содержание не ищется."""
    card = filler("карточка", 80)
    distillate = filler("дистиллят", 60) + " " + card + " " + filler("хвост", 40)
    records = containment_records(card, distillate)

    kept, stats = dedup_mod.containment_dedup(
        records, extract_blocks=lambda text: [text.split(" ")[0]]
    )

    assert {record[0] for record in kept} == {"k-1", "d-1"}
    assert stats.dropped == 0


def test_tc3_container_is_never_dropped():
    """Контейнер не снимается контейнментом, даже если он сам вложен в другого."""
    inner = filler("блок", 80)
    outer = filler("обёртка", 60) + " " + inner + " " + filler("хвост", 40)
    records = [("d-2", "D", outer), ("d-1", "D", inner)]

    kept, stats = dedup_mod.containment_dedup(records)

    assert [record[0] for record in kept] == ["d-2", "d-1"]
    assert stats.dropped == 0
    assert stats.checked == 0
