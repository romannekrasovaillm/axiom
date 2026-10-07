#!/usr/bin/env python3
"""C-047 — согласованность чисел ADR-009 и ``net/config.json`` (ENVIRONMENT-V1 §9).

Инвариант AD-9 требует, чтобы механика длинного контекста была объявлена
*декларативно*: параметром ``net/config.json``. ADR-009 при этом называет те же
числа прозой («n_win = 128», «``top_k`` (512 …)», «``mla_latent_dim: 512``»).
Пока это так, число существует в ДВУХ местах, и они могут разойтись незаметно
для структурных правил: ``must_contain`` проверяет наличие фразы, а не её
значение. Тогда решение говорит одно, код делает другое, и оба формально
зелёные.

Страж сверяет стороны по ДЕКЛАРАТИВНОЙ карте (``tools/adr_config_map.yaml``):
для каждого соответствия берётся паттерн ADR с группой-числом и поле конфига;
все вхождения паттерна в тексте ADR обязаны нести одно и то же число, и оно
обязано совпадать со значением поля в конфиге.

Договор вердикта — **fail-closed**:

* ``0`` PASS — каждое соответствие карты разрешено и согласовано;
* ``1`` FAIL — есть находки, в том числе класса ``adr-unverifiable``
  (паттерн карты не нашёлся в тексте ADR, поле конфига отсутствует, карта не
  читается). Непроверяемое соответствие — НЕ pass: дрейф текста обязан ломать
  страж громко, иначе переписанная формулировка числа молча снимает контроль.
  Третьего кода нет намеренно: «сверять не с чем» — это тоже находка.

Классы находок:

* ``adr-config-mismatch``  — число в ADR ≠ число в конфиге (рассинхрон сторон);
* ``adr-unverifiable``     — формулировка/файл ADR не найдены, число не
  извлекается (паттерн карты устарел или текст removed);
* ``config-unverifiable``  — поля нет в конфиге или его значение не целое;
* ``map-unverifiable``     — карта не найдена/не разбирается/структурно дефектна.

Запуск::

    python3 tools/check_adr_config_consistency.py [--case DIR] [--map PATH]
                                                  [--json] [--quiet]

Карта по умолчанию ищется в кейсе (``<case>/tools/adr_config_map.yaml`` — карта
едет вместе со снапшотом), затем рядом со скриптом. Скрипт stdlib-only: в среде
гейта внешних зависимостей нет, а падать на импорте страж не имеет права.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
CASE_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import adr_config_map as acm  # noqa: E402  (путь добавляется выше)

CLASS_MISMATCH = "adr-config-mismatch"
CLASS_ADR_UNVERIFIABLE = "adr-unverifiable"
CLASS_CONFIG_UNVERIFIABLE = "config-unverifiable"
CLASS_MAP_UNVERIFIABLE = "map-unverifiable"

_MISSING = object()


def _finding(cls: str, mapping: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"class": cls, "mapping": mapping, "message": message, **extra}


def _lookup(obj: Any, dotted: str) -> Any:
    """Значение по точечному пути ``a.b.c``; ``_MISSING``, если пути нет.

    Числовые сегменты адресуют индексы списков (``pins.0.threshold``) —
    ADR-032: пин KPI живёт массивом в ``evidence/kpi-pins.json``.
    """
    cur = obj
    for part in dotted.split("."):
        if part.isdigit():
            if not isinstance(cur, list) or int(part) >= len(cur):
                return _MISSING
            cur = cur[int(part)]
            continue
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _extract_adr_values(text: str, pattern: str) -> tuple[list[int], str | None]:
    """Все числа группы 1 по паттерну. Второй элемент — причина отказа (или None)."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return [], f"паттерн карты не компилируется: {exc}"
    values: list[int] = []
    for match in rx.finditer(text):
        raw = match.group(1)
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            return [], f"группа паттерна не число: {raw!r}"
    return values, None


def _default_map(case_dir: Path) -> Path:
    """Карта кейса, если она есть; иначе — карта рядом со скриптом."""
    local = case_dir / "tools" / "adr_config_map.yaml"
    return local if local.is_file() else acm.DEFAULT_MAP


def evaluate(case_dir: Path | str = CASE_DIR, map_path: Path | str | None = None) -> tuple[int, dict[str, Any]]:
    """Прогон стража. Возвращает ``(код возврата, JSON-отчёт)``.

    ``0`` — все соответствия карты согласованы; ``1`` — есть находки.
    """
    case_dir = Path(case_dir)
    path = Path(map_path) if map_path is not None else _default_map(case_dir)
    report: dict[str, Any] = {
        "passed": False,
        "case": case_dir.as_posix(),
        "map": path.as_posix(),
        "findings": [],
        "checked": 0,
        "summary": "",
    }

    # ── карта: нечитаемая карта — не «сверять не с чем», а находка ───────────
    try:
        spec = acm.load(path)
    except acm.MapError as exc:
        report["findings"].append(_finding(CLASS_MAP_UNVERIFIABLE, "*", str(exc)))
        report["summary"] = f"C-047: карта не читается — {exc}"
        return 1, report

    problems = acm.validate(spec)
    if problems:
        for problem in problems:
            report["findings"].append(_finding(CLASS_MAP_UNVERIFIABLE, "*", problem))
        report["summary"] = f"C-047: карта структурно дефектна — находок {len(problems)}"
        return 1, report

    mappings = acm.get_mappings(spec)
    config_path = case_dir / acm.config_rel(spec)

    # ── файлы сторон: их отсутствие — находка на каждом соответствии ────────
    adr_files: list[str] = []
    adr_error: str | None = None
    try:
        adr_files = acm.resolve_adr(case_dir, spec)
        if len(adr_files) != 1:
            adr_error = (
                f"glob {spec.get('adr_glob')!r} разрешается в {len(adr_files)} файлов "
                f"({', '.join(adr_files) or 'пусто'}) — сверять не с чем"
            )
    except acm.MapError as exc:
        adr_error = str(exc)
    report["adr"] = adr_files[0] if len(adr_files) == 1 else None

    if adr_error is None:
        adr_text = (case_dir / adr_files[0]).read_text(encoding="utf-8")
    else:
        adr_text = ""

    config_obj: Any = _MISSING
    config_error: str | None = None
    if not config_path.is_file():
        config_error = f"конфиг не найден: {config_path.as_posix()}"
    else:
        try:
            config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            config_error = f"конфиг не разбирается: {exc}"
    report["config"] = acm.config_rel(spec)

    # ── сверка ───────────────────────────────────────────────────────────────
    for entry in mappings:
        mid = entry["id"]
        field = entry["config_field"]
        report["checked"] += 1

        if adr_error is not None:
            report["findings"].append(
                _finding(CLASS_ADR_UNVERIFIABLE, mid, adr_error, adr_pattern=entry["adr_pattern"],
                         config_field=field)
            )
            continue

        values, why = _extract_adr_values(adr_text, entry["adr_pattern"])
        if why is not None or not values:
            report["findings"].append(
                _finding(
                    CLASS_ADR_UNVERIFIABLE,
                    mid,
                    why or f"паттерн {entry['adr_pattern']!r} не найден в {adr_files[0]} — "
                           "формулировка числа в ADR изменилась, карту пора править",
                    adr_file=adr_files[0],
                    adr_pattern=entry["adr_pattern"],
                    config_field=field,
                )
            )
            continue

        if config_error is not None:
            report["findings"].append(
                _finding(CLASS_CONFIG_UNVERIFIABLE, mid, config_error, config_field=field,
                         adr_file=adr_files[0], adr_value=values[0])
            )
            continue

        cfg_value = _lookup(config_obj, field)
        if cfg_value is _MISSING:
            report["findings"].append(
                _finding(CLASS_CONFIG_UNVERIFIABLE, mid, f"поля «{field}» нет в конфиге",
                         config_field=field, adr_file=adr_files[0], adr_value=values[0])
            )
            continue
        if isinstance(cfg_value, bool) or not isinstance(cfg_value, int):
            report["findings"].append(
                _finding(CLASS_CONFIG_UNVERIFIABLE, mid,
                         f"значение поля «{field}» не целое: {cfg_value!r}",
                         config_field=field, config_value=cfg_value, adr_value=values[0])
            )
            continue

        if len(set(values)) > 1:
            report["findings"].append(
                _finding(
                    CLASS_MISMATCH,
                    mid,
                    f"формулировки ADR противоречат друг другу: {values} "
                    f"(вхождений паттерна {entry['adr_pattern']!r} — {len(values)})",
                    adr_file=adr_files[0], adr_pattern=entry["adr_pattern"],
                    adr_value=values[0], config_field=field, config_value=cfg_value,
                )
            )
            continue

        if values[0] != cfg_value:
            report["findings"].append(
                _finding(
                    CLASS_MISMATCH,
                    mid,
                    f"ADR говорит {values[0]}, config.{field} = {cfg_value}",
                    adr_file=adr_files[0], adr_pattern=entry["adr_pattern"],
                    adr_value=values[0], config_field=field, config_value=cfg_value,
                )
            )

    report["passed"] = not report["findings"]
    if report["passed"]:
        report["summary"] = (
            f"C-047: {report['checked']} соответствий ADR↔config согласованы "
            f"({report['adr']})"
        )
    else:
        classes = sorted({f["class"] for f in report["findings"]})
        report["summary"] = (
            f"C-047: {report['checked']} соответствий, находок {len(report['findings'])} "
            f"({', '.join(classes)})"
        )
    return (0 if report["passed"] else 1), report


def _format_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for f in report["findings"]:
        lines.append(f"FAIL [{f['class']}] {f['mapping']}: {f['message']}")
    if report["passed"]:
        lines.append(f"PASS {report['summary']}")
    else:
        lines.append(f"FAIL {report['summary']}")
    return "\n".join(lines)


def _selftest() -> int:
    """Мутантный самотест стража (канон C-043/44/45): синтетический мини-кейс.

    Строит tmp-кейс (ADR + config + карта), проверяет: (1) согласованное
    состояние зелёное; (2) дрейф конфиг-стороны красит; (3) дрейф ADR-стороны
    красит; (4) исчезновение паттерна из ADR = adr-unverifiable (fail-closed,
    не pass); (5) точечный путь с индексом списка (pins.0.x) разрешается.
    Возвращает 0, если все пять мутантов пойманы, иначе 1.
    """
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as td:
        case = _Path(td)
        (case / "docs" / "adr").mkdir(parents=True)
        (case / "tools").mkdir()
        (case / "evidence").mkdir()
        adr = case / "docs" / "adr" / "ADR-777-selftest.md"
        adr.write_text(
            "# ADR-777 selftest\n\nОкно n_win = 128. Порог ≥800 ток/с.\n",
            encoding="utf-8",
        )
        cfg = case / "evidence" / "pins.json"
        cfg.write_text('{"pins": [{"threshold": 800}], "swa_window": 128}', encoding="utf-8")
        mp = case / "tools" / "map.yaml"
        mp.write_text(
            'version: 1\n'
            'adr_glob: "docs/adr/ADR-777-selftest.md"\n'
            'config: "evidence/pins.json"\n'
            'mappings:\n'
            '  - id: win\n'
            '    note: "n_win"\n'
            '    adr_pattern: "n_win = (\\\\d+)"\n'
            '    config_field: "swa_window"\n'
            '    drift: true\n'
            '  - id: thr\n'
            '    note: "threshold"\n'
            '    adr_pattern: "≥(\\\\d+) ток/с"\n'
            '    config_field: "pins.0.threshold"\n'
            '    drift: true\n',
            encoding="utf-8",
        )

        def _run() -> tuple[int, str]:
            code, report = evaluate(str(case), str(mp))
            return code, _format_human(report)

        # (1) согласованное состояние — зелёное
        code, out = _run()
        if code != 0:
            print(f"SELFTEST FAIL (1): согласованное состояние красное:\n{out}")
            return 1
        # (2) дрейф конфиг-стороны (128→256)
        cfg.write_text('{"pins": [{"threshold": 800}], "swa_window": 256}', encoding="utf-8")
        code, out = _run()
        if code == 0:
            print("SELFTEST FAIL (2): дрейф конфига не пойман")
            return 1
        # (3) дрейф ADR-стороны (128→129)
        cfg.write_text('{"pins": [{"threshold": 800}], "swa_window": 128}', encoding="utf-8")
        adr.write_text("# ADR-777 selftest\n\nОкно n_win = 129. Порог ≥800 ток/с.\n", encoding="utf-8")
        code, out = _run()
        if code == 0:
            print("SELFTEST FAIL (3): дрейф ADR не пойман")
            return 1
        # (4) паттерн исчез из ADR → adr-unverifiable, fail-closed
        adr.write_text("# ADR-777 selftest\n\nТекст без числа.\n", encoding="utf-8")
        code, out = _run()
        if code == 0 or "adr-unverifiable" not in out:
            print(f"SELFTEST FAIL (4): исчезновение паттерна не adr-unverifiable:\n{out}")
            return 1
        # (5) list-путь ловится и красится (800→799)
        adr.write_text("# ADR-777 selftest\n\nОкно n_win = 128. Порог ≥800 ток/с.\n", encoding="utf-8")
        cfg.write_text('{"pins": [{"threshold": 799}], "swa_window": 128}', encoding="utf-8")
        code, out = _run()
        if code == 0:
            print("SELFTEST FAIL (5): дрейф pins.0.threshold не пойман")
            return 1
    print("SELFTEST PASS: 5/5 (зелёный базовый, дрейф config, дрейф ADR, fail-closed паттерна, list-путь)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-047: согласованность чисел ADR и конфига/пинов (декларативные карты)")
    parser.add_argument("--case", default=str(CASE_DIR), help="корень кейса (по умолчанию — рядом со скриптом)")
    parser.add_argument("--map", default=None, help="карта соответствий (по умолчанию — карта кейса/скрипта)")
    parser.add_argument("--json", action="store_true", help="печатать JSON-отчёт вместо строк")
    parser.add_argument("--quiet", action="store_true", help="только вердикт (строка-сводка)")
    parser.add_argument("--selftest", action="store_true",
                        help="мутантный самотест на синтетическом мини-кейсе (канон C-043/44/45)")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    code, report = evaluate(args.case, args.map)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.quiet:
        print(f"{'PASS' if report['passed'] else 'FAIL'} {report['summary']}")
    else:
        print(_format_human(report))
    return code


if __name__ == "__main__":
    sys.exit(main())
