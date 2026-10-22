#!/usr/bin/env python3
"""C-035 — декларативность механики длинного контекста: сверка config ↔ код (AD-9).

Инвариант AD-9 требует, чтобы механика длинного контекста вносилась
*декларативно* — параметром ``net/config.json`` и строкой в ``deviations``, — а
не правкой кода по месту.  Страж проверяет именно это соответствие, а не
наличие надписи (ADR-011, AD-10):

1. **Объявление** (``net/config.json``): объект ``mla_block_merge`` с
   ``enabled`` (bool) и ``block`` (целое ≥ 1); при включённом механизме окно
   обязано покрывать блок (``swa_window >= block``) — иначе хвост блока до
   ``block - 1`` записей остаётся без внимания (ADR-018);
2. **Схема** (``net/config.py``): поле объявлено в ``ModelConfig``, читатель
   ``declared_block_merge`` существует и действительно читает это поле;
3. **Потребитель** (``net/attn_sparse.py``): параметр ``block_merge`` у
   ``sparse_union_attention``, значение параметра используется телом функции, и
   вызов читателя присутствует — объявление доходит до внимания, а не лежит
   рядом;
4. **Проба читателя** (поведенческая): включённая декларация читается как
   ширина блока, выключенная/отсутствующая — как ``0``; сама проверяемая
   декларация читается тем же значением, что в ней записано.  Проба идёт через
   ``net.config`` — модуль схемы не тянет численных зависимостей, поэтому страж
   запускается обычным ``python3`` без jax;
5. **Остальные механики AD-9** (``swa_window``, ``mla_top_k``, шаринг
   KDA-проекций, FP4 латентного KV, seed индексатора, параметры пула и раскладка
   режимов): объявлены и читаются кодом ``net/`` вне схемы — объявленное поле,
   которое никто не читает, — это декларация без механизма;
6. **Решение названо**: строка ``deviations`` со ссылкой на ADR-018 и на
   источник заимствования (Step-5-Preview / StepFun).

Входы (по умолчанию — каталог кейса рядом со скриптом, переопределяются):

* ``--config`` — файл декларации (по умолчанию ``net/config.json`` кейса);
* ``--net-dir`` — каталог кода ``net/`` (по умолчанию ``net/`` кейса);
* ``--quiet`` — печатать только вердикт (stderr).

Запуск::

    python3 tools/check_declarative_context.py [--config PATH] [--net-dir DIR]

Коды возврата:
  * ``0``  — PASS: декларация и код согласованы;
  * ``1``  — FAIL: рассогласование (объявлено, но не читается / прочитано, но не
             объявлено / объявление неполно или противоречиво);
  * ``2``  — NOT-VERIFIED: сверять не с чем или проба читателя не запускается —
             ложный PASS запрещён.

Скрипт — stdlib-only (никакого PyYAML и никакого jax): в среде гейта внешних
зависимостей нет, а падать на импорте страж не имеет права.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path

#: Каталог кейса — рядом со скриптом (``tools/`` → корень), а не текущий каталог:
#: гейт запускается и не из корня, и вердикт от этого зависеть не должен (AD-11).
CASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CASE_DIR / "net" / "config.json"
DEFAULT_NET_DIR = CASE_DIR / "net"

#: Модуль схемы декларации и модуль, который её потребляет (ADR-018).
SCHEMA_MODULE = "config.py"
CONSUMER_MODULE = "attn_sparse.py"

#: Поле-переключатель block-wise token merging и его кодовая проекция.
BLOCK_MERGE_FIELD = "mla_block_merge"
BLOCK_MERGE_READER = "declared_block_merge"
BLOCK_MERGE_PARAM = "block_merge"
CONSUMER_FUNCTION = "sparse_union_attention"

#: Механики AD-9, которые обязаны быть объявлены и прочитаны кодом ``net/``.
#: Поле блока-слияния проверяется отдельно и строже (читатель + потребитель).
DECLARED_MECHANISMS = (
    "swa_window",
    "mla_top_k",
    "swa_share_kda_projections",
    "qat_kv_enabled",
    "indexer_seed",
    "mla_pool_block",
    "mla_pool_size",
    "mla_layer_modes",
)

#: Строка deviations обязана назвать решение и источник заимствования.
DEVIATION_DECISION = "ADR-018"
DEVIATION_SOURCES = ("Step-5", "StepFun")

#: Функции-аксессоры отображений, чей первый аргумент считается чтением ключа.
_MAPPING_READERS = ("get", "pop", "setdefault")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NOT_VERIFIED = 2


# ---------------------------------------------------------------------------
# разбор исходников (AST): что модуль читает и что объявляет
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """``NAME = "литерал"`` уровня модуля — один уровень разрешения имён.

    Читатель берёт ключ из константы (``payload.get(BLOCK_MERGE_KEY)``), поэтому
    литерал в коде надо разрешить, иначе чтение поля выглядело бы как чтение
    имени.  Глубже одного уровня не разворачиваем: этого достаточно для
    объявленного читателя и не превращает проверку в интерпретатор.
    """
    constants: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _key_strings(node: ast.expr, constants: dict[str, str]) -> set[str]:
    """Строковые ключи выражения: литерал или разрешённая константа."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and node.id in constants:
        return {constants[node.id]}
    return set()


def code_reads(path: Path) -> set[str]:
    """Имена, которые модуль читает: атрибуты, ключи отображений, имена.

    ``cfg.swa_window`` даёт ``swa_window``; ``payload.get("mla_block_merge")`` и
    ``payload.get(BLOCK_MERGE_KEY)`` — ``mla_block_merge``.  Комментарии и
    докстроки в разбор не попадают: упоминание в прозе чтением не считается
    (ADR-011: страж проверяет поведение, а не прозу).
    """
    tree = _parse(path)
    constants = _module_string_constants(tree)
    reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            reads.add(node.attr)
        elif isinstance(node, ast.Subscript):
            reads |= _key_strings(node.slice, constants)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MAPPING_READERS
            and node.args
        ):
            reads |= _key_strings(node.args[0], constants)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads.add(node.id)
    return reads


def dataclass_fields(path: Path, class_name: str) -> set[str]:
    """Поля dataclass'а ``class_name`` (аннотированные атрибуты класса)."""
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                for target in [statement.target]
            }
    return set()


def functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def parameter_names(function: ast.FunctionDef) -> set[str]:
    args = function.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def names_used(function: ast.FunctionDef) -> set[str]:
    """Имена, которые тело функции читает (без самой подписи и её аннотаций)."""
    used: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    return used


def calls_name(tree: ast.Module, name: str) -> bool:
    """Есть ли в модуле вызов ``name(...)`` или ``модуль.name(...)``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


# ---------------------------------------------------------------------------
# проба читателя (поведенческая, без jax)
# ---------------------------------------------------------------------------


def probe_reader(config: Path, declared_width: int | None) -> tuple[int, list[str], str]:
    """Прогон читателя на живом модуле схемы и на фикстурах.

    Возвращает ``(код, находки, сообщение)``.  Код ``EXIT_NOT_VERIFIED`` — если
    читателя не удалось поднять (тогда сверка не состоялась, а не «зелено»);
    ``EXIT_FAIL`` — если проба поднялась, но читает не то, что объявлено.
    """
    if str(CASE_DIR) not in sys.path:
        sys.path.insert(0, str(CASE_DIR))
    try:  # noqa: PLC0415 — проба обязана идти по живому модулю кейса
        from net.config import declared_block_merge
    except Exception as exc:  # noqa: BLE001 — среда без модуля кейса: не PASS
        return (
            EXIT_NOT_VERIFIED,
            [],
            "NOT-VERIFIED: читатель net.config.declared_block_merge не поднялся "
            f"({type(exc).__name__}: {exc}) — сверка декларации с кодом не выполнена",
        )

    findings: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = (
            ("enabled", {"mla_block_merge": {"enabled": True, "block": 16}}, 16),
            ("disabled", {"mla_block_merge": {"enabled": False, "block": 16}}, 0),
            ("absent", {"swa_window": 128}, 0),
        )
        for name, payload, expected in fixtures:
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            got = declared_block_merge(path)
            if got != expected:
                findings.append(
                    f"проба читателя ({name}): ожидалось {expected}, получено {got!r}"
                )
        if declared_block_merge(Path(tmp) / "missing.json") != 0:
            findings.append("проба читателя: отсутствующий файл прочитан не как 0")

    if declared_width is not None and declared_block_merge(config) != declared_width:
        findings.append(
            "проба читателя: объявленная декларация читается как "
            f"{declared_block_merge(config)!r}, а в ней записано {declared_width}"
        )
    return (EXIT_FAIL if findings else EXIT_PASS), findings, ""


# ---------------------------------------------------------------------------
# сверка
# ---------------------------------------------------------------------------


def _declared_width(payload: dict) -> int | None:
    """Ширина блока, если объявление полное и валидное, иначе ``None``."""
    declared = payload.get(BLOCK_MERGE_FIELD)
    if not isinstance(declared, dict):
        return None
    enabled, block = declared.get("enabled"), declared.get("block")
    if not isinstance(enabled, bool):
        return None
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        return None
    return block if enabled else 0


def _check_declaration(payload: dict, findings: list[str]) -> None:
    """(1) объявление полное, валидное и согласовано с окном."""
    declared = payload.get(BLOCK_MERGE_FIELD)
    if not isinstance(declared, dict):
        findings.append(
            f"{BLOCK_MERGE_FIELD}: нет объекта в декларации — механизм ADR-018 "
            "не объявлен, хотя код его читает"
        )
        return
    enabled, block = declared.get("enabled"), declared.get("block")
    if not isinstance(enabled, bool):
        findings.append(f"{BLOCK_MERGE_FIELD}.enabled: ожидалось true/false, получено {enabled!r}")
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        findings.append(
            f"{BLOCK_MERGE_FIELD}.block: ожидалось целое >= 1 (записей в блоке), получено {block!r}"
        )
        return
    if enabled is True:
        window = payload.get("swa_window")
        if not isinstance(window, int) or window < block:
            findings.append(
                f"окно не покрывает блок: swa_window={window!r} < {BLOCK_MERGE_FIELD}.block={block} — "
                "хвост блока (до block - 1 записей) остался бы без внимания (ADR-018)"
            )


def _check_schema(net_dir: Path, findings: list[str]) -> None:
    """(2) схема: поле объявлено в ModelConfig, читатель читает именно его."""
    schema = net_dir / SCHEMA_MODULE
    if not schema.is_file():
        findings.append(f"нет {schema.name} — схема декларации не найдена")
        return
    if BLOCK_MERGE_FIELD not in dataclass_fields(schema, "ModelConfig"):
        findings.append(f"{SCHEMA_MODULE}: поле {BLOCK_MERGE_FIELD} не объявлено в ModelConfig")
    if BLOCK_MERGE_READER not in functions(_parse(schema)):
        findings.append(f"{SCHEMA_MODULE}: нет читателя {BLOCK_MERGE_READER}")
    elif BLOCK_MERGE_FIELD not in code_reads(schema):
        findings.append(
            f"{SCHEMA_MODULE}: читатель не читает поле {BLOCK_MERGE_FIELD} "
            "(ни ключа, ни литерала в коде модуля)"
        )


def _check_consumer(net_dir: Path, findings: list[str]) -> None:
    """(3) потребитель: параметр есть, значение используется, читатель вызван."""
    consumer = net_dir / CONSUMER_MODULE
    if not consumer.is_file():
        findings.append(f"нет {consumer.name} — потребитель декларации не найден")
        return
    tree = _parse(consumer)
    function = functions(tree).get(CONSUMER_FUNCTION)
    if function is None:
        findings.append(f"{CONSUMER_MODULE}: нет функции {CONSUMER_FUNCTION}")
    else:
        if BLOCK_MERGE_PARAM not in parameter_names(function):
            findings.append(
                f"{CONSUMER_MODULE}:{CONSUMER_FUNCTION} не принимает {BLOCK_MERGE_PARAM} — "
                "декларации некуда дойти"
            )
        elif BLOCK_MERGE_PARAM not in names_used(function):
            findings.append(
                f"{CONSUMER_MODULE}:{CONSUMER_FUNCTION} принимает {BLOCK_MERGE_PARAM}, "
                "но не использует его — параметр-заглушка"
            )
    if not calls_name(tree, BLOCK_MERGE_READER):
        findings.append(
            f"{CONSUMER_MODULE} не вызывает {BLOCK_MERGE_READER}: декларация не читается "
            "вниманием (значение берётся не из net/config.json)"
        )


def _check_deviations(payload: dict, findings: list[str]) -> None:
    """(6) решение названо: строка deviations со ссылкой на ADR-018 и источник."""
    deviations = payload.get("deviations")
    if not isinstance(deviations, list):
        findings.append("нет секции deviations — источник заимствования не назван")
        return
    lines = [
        line for line in deviations if isinstance(line, str) and DEVIATION_DECISION in line
    ]
    if not lines:
        findings.append(
            f"нет строки deviations со ссылкой на {DEVIATION_DECISION} — "
            "механика внесена молча"
        )
        return
    line = lines[0]
    if BLOCK_MERGE_FIELD not in line:
        findings.append(f"строка deviations не называет поле {BLOCK_MERGE_FIELD}")
    if not any(source in line for source in DEVIATION_SOURCES):
        findings.append(
            "строка deviations не называет источник заимствования "
            f"({'/'.join(DEVIATION_SOURCES)})"
        )


def _mechanism_table(
    payload: dict, net_dir: Path, findings: list[str]
) -> list[tuple[str, str, str]]:
    """(5) остальные механики AD-9: объявлены и читаются кодом вне схемы."""
    reads: dict[str, set[str]] = {}
    for path in sorted(net_dir.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        reads[path.name] = code_reads(path)

    table: list[tuple[str, str, str]] = []
    for field in DECLARED_MECHANISMS:
        declared = "да" if field in payload else "нет"
        readers = sorted(
            name for name, names in reads.items() if field in names and name != SCHEMA_MODULE
        )
        table.append((field, declared, ", ".join(readers) if readers else "—"))
        if field not in payload:
            findings.append(f"{field}: механика AD-9 не объявлена в декларации")
        elif not readers:
            findings.append(
                f"{field}: объявлено, но не читается кодом net/ вне {SCHEMA_MODULE} — "
                "декларация без механизма"
            )
    return table


def evaluate(
    config: Path | None = None, net_dir: Path | None = None
) -> tuple[int, list[str]]:
    """Сверяет декларацию и код.  Возвращает ``(код, строки вывода)``."""
    config = Path(config) if config is not None else DEFAULT_CONFIG
    net_dir = Path(net_dir) if net_dir is not None else DEFAULT_NET_DIR
    lines: list[str] = []

    if not config.is_file():
        lines.append(
            f"NOT-VERIFIED: декларация {config} не найдена — сверять не с чем "
            "(ложный PASS запрещён)"
        )
        return EXIT_NOT_VERIFIED, lines
    if not net_dir.is_dir():
        lines.append(
            f"NOT-VERIFIED: каталог кода {net_dir} не найден — сверять не с чем "
            "(ложный PASS запрещён)"
        )
        return EXIT_NOT_VERIFIED, lines
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except ValueError as exc:
        lines.append(f"NOT-VERIFIED: декларация {config} не читается как JSON — {exc}")
        return EXIT_NOT_VERIFIED, lines
    if not isinstance(payload, dict):
        lines.append(f"NOT-VERIFIED: декларация {config} — не объект JSON")
        return EXIT_NOT_VERIFIED, lines

    findings: list[str] = []
    _check_declaration(payload, findings)
    _check_schema(net_dir, findings)
    _check_consumer(net_dir, findings)
    table = _mechanism_table(payload, net_dir, findings)
    _check_deviations(payload, findings)

    probe_code, probe_findings, probe_note = probe_reader(config, _declared_width(payload))
    findings.extend(probe_findings)

    lines.append(f"декларация: {config}")
    lines.append(f"код: {net_dir}")
    lines.append(f"механики AD-9 (поле → объявлено → читается в коде):")
    for field, declared, readers in table:
        lines.append(f"  {field:<26} {declared:<4} {readers}")
    lines.append(
        f"  {BLOCK_MERGE_FIELD:<26} {'да' if BLOCK_MERGE_FIELD in payload else 'нет':<4} "
        f"{SCHEMA_MODULE} ({BLOCK_MERGE_READER}) → {CONSUMER_MODULE} ({BLOCK_MERGE_PARAM})"
    )
    if probe_note:
        lines.append(probe_note)

    if probe_code == EXIT_NOT_VERIFIED:
        lines.append("")
        lines.append("Итог: NOT-VERIFIED")
        return EXIT_NOT_VERIFIED, lines

    if findings:
        lines.append("")
        lines.append(f"Нарушения ({len(findings)}):")
        lines.extend(f"  - {finding}" for finding in findings)
        lines.append("")
        lines.append("Итог: FAIL")
        return EXIT_FAIL, lines

    lines.append("")
    lines.append(
        "Итог: PASS — объявленное читается кодом: mla_block_merge через "
        f"{SCHEMA_MODULE} ({BLOCK_MERGE_READER}) потребляется {CONSUMER_MODULE} "
        f"({BLOCK_MERGE_PARAM}), остальные механики AD-9 читаются слоями"
    )
    return EXIT_PASS, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-035: декларативность механики длинного контекста (AD-9, ADR-018)",
    )
    parser.add_argument("--config", type=Path, default=None, help="файл декларации")
    parser.add_argument("--net-dir", type=Path, default=None, help="каталог кода net/")
    parser.add_argument(
        "--quiet", action="store_true", help="печатать только вердикт (stderr)"
    )
    args = parser.parse_args(argv)

    code, lines = evaluate(args.config, args.net_dir)
    if not args.quiet:
        for line in lines[:-1]:
            print(line)
    # Вердикт — в stderr: его читает гейт, stdout остаётся диагностике.
    print(lines[-1], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
