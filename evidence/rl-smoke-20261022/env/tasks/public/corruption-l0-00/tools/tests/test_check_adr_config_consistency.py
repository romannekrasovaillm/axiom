"""T-C047 — страж согласованности «число в ADR-009 ↔ поле net/config.json».

Страж (`tools/check_adr_config_consistency.py`) сверяет две стороны одного
решения по декларативной карте (`tools/adr_config_map.yaml`) и подчиняется
договору **fail-closed**:

* ``0`` — все соответствия карты разрешены и согласованы;
* ``1`` — есть находки, включая ``adr-unverifiable`` (формулировка карты не
  нашлась в тексте ADR) и ``map-unverifiable`` (карта не читается/дефектна).

Тесты фиксируют четыре состояния стража и его связку с мутатором порчи:
порча числом ловится (``adr-config-mismatch``), откат снимает находку,
``claim_inversion`` стражем НЕ ловится — это объявленный класс модель-детекции
(ENVIRONMENT-V1 §9), и проверка этого факта здесь документирует границу.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = TOOLS_DIR.parent
for _path in (TOOLS_DIR, CASE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import adr_config_map as acm  # noqa: E402
import check_adr_config_consistency as guard  # noqa: E402
from env import corruption  # noqa: E402
from env.util import copy_case_snapshot  # noqa: E402

ADR_GLOB = "docs/adr/ADR-009-*.md"

#: Поля карты, которые мутатор обязан ловить стражем (связка §9).
C047_ATOMS = ("numeric_drift", "term_swap", "config_drift")


def _adr(case: Path) -> Path:
    found = list(case.glob(ADR_GLOB))
    assert len(found) == 1, found
    return found[0]


def _mini_case(tmp_path: Path, name: str = "case") -> Path:
    """Мини-кейс из двух настоящих сторон: ADR-009 и net/config.json.

    Карта при этом берётся штатная (кейс своей не несёт → страж падает в
    fallback рядом со скриптом): тест проверяет вердикт, а не доставку карты.
    """
    root = tmp_path / name
    (root / "docs" / "adr").mkdir(parents=True)
    (root / "net").mkdir()
    shutil.copy2(_adr(CASE_DIR), root / ADR_GLOB.replace("*", _adr(CASE_DIR).name.split("ADR-009-", 1)[1]))
    shutil.copy2(CASE_DIR / "net" / "config.json", root / "net" / "config.json")
    return root


def _write_config(case: Path, payload: dict) -> None:
    (case / "net" / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _config(case: Path) -> dict:
    return json.loads((case / "net" / "config.json").read_text(encoding="utf-8"))


def _classes(report: dict) -> set[str]:
    return {f["class"] for f in report["findings"]}


# ── страж на каноническом кейсе ──────────────────────────────────────────────

def test_canonical_case_passes():
    """ADR-009 и net/config.json кейса согласованы — страж зелёный."""
    code, report = guard.evaluate(CASE_DIR)
    assert code == 0, report["findings"]
    assert report["checked"] == len(acm.get_mappings(acm.load(TOOLS_DIR / "adr_config_map.yaml")))
    assert report["findings"] == []
    assert report["adr"].endswith(".md") and report["config"] == "net/config.json"


def test_shipped_map_is_wellformed():
    """Карта проходит собственную структурную валидацию (fail-closed на дрейф)."""
    spec = acm.load(TOOLS_DIR / "adr_config_map.yaml")
    assert acm.validate(spec) == []
    assert acm.resolve_adr(CASE_DIR, spec) == [p.relative_to(CASE_DIR).as_posix()
                                               for p in CASE_DIR.glob(ADR_GLOB)]


# ── находки: рассинхрон и «сверять не с чем» ─────────────────────────────────

def test_mutant_number_is_mismatch(tmp_path):
    """Мутант числа в ADR → adr-config-mismatch с обеими сторонами в отчёте."""
    case = _mini_case(tmp_path)
    adr = _adr(case)
    text = adr.read_text(encoding="utf-8")
    assert "`mla_latent_dim: 512`" in text
    adr.write_text(text.replace("`mla_latent_dim: 512`", "`mla_latent_dim: 1024`"), encoding="utf-8")

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["mapping"] == "mla_latent_dim")
    assert finding["class"] == "adr-config-mismatch"
    assert finding["adr_value"] == 1024 and finding["config_value"] == 512
    assert finding["config_field"] == "mla_latent_dim"


def test_mutant_config_number_is_mismatch(tmp_path):
    """Мутант числа в config → тот же класс находки, сторона — config."""
    case = _mini_case(tmp_path)
    config = _config(case)
    config["mla_top_k"] = 1024
    _write_config(case, config)

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["config_field"] == "mla_top_k")
    assert finding["class"] == "adr-config-mismatch"
    assert finding["adr_value"] == 512 and finding["config_value"] == 1024


def test_removed_formulation_is_unverifiable(tmp_path):
    """Формулировки нет в тексте → adr-unverifiable (fail-closed), а не pass."""
    case = _mini_case(tmp_path)
    adr = _adr(case)
    text = adr.read_text(encoding="utf-8")
    adr.write_text(text.replace("n_win = 128", "окно фиксировано"), encoding="utf-8")

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["mapping"] == "swa_window")
    assert finding["class"] == "adr-unverifiable"
    assert "не найден" in finding["message"]


def test_disagreeing_formulations_are_mismatch(tmp_path):
    """Одна формулировка изменена, вторая осталась → ADR противоречит сам себе."""
    case = _mini_case(tmp_path)
    adr = _adr(case)
    text = adr.read_text(encoding="utf-8")
    # «n_win = 128» встречается дважды; правим только первое вхождение
    adr.write_text(text.replace("n_win = 128", "n_win = 256", 1), encoding="utf-8")

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["mapping"] == "swa_window")
    assert finding["class"] == "adr-config-mismatch"
    assert "противоречат" in finding["message"]


def test_missing_config_field_is_unverifiable(tmp_path):
    """Поля нет в конфиге → config-unverifiable (не «просто pass»)."""
    case = _mini_case(tmp_path)
    config = _config(case)
    del config["mla_index_dim"]
    _write_config(case, config)

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["mapping"] == "mla_index_dim")
    assert finding["class"] == "config-unverifiable"
    assert "нет в конфиге" in finding["message"]


def test_non_integer_config_value_is_unverifiable(tmp_path):
    """Нечисловое значение поля — сверять нечего, ложный pass запрещён."""
    case = _mini_case(tmp_path)
    config = _config(case)
    config["swa_window"] = "128"
    _write_config(case, config)

    code, report = guard.evaluate(case)
    assert code == 1
    finding = next(f for f in report["findings"] if f["mapping"] == "swa_window")
    assert finding["class"] == "config-unverifiable"
    assert "не целое" in finding["message"]


def test_missing_adr_is_unverifiable(tmp_path):
    """Файла решения нет → каждое соответствие неверифицируемо (fail-closed)."""
    case = _mini_case(tmp_path)
    _adr(case).unlink()

    code, report = guard.evaluate(case)
    assert code == 1
    assert _classes(report) == {"adr-unverifiable"}
    assert report["adr"] is None


def test_missing_map_is_not_pass(tmp_path):
    """Нечитаемая карта — находка, а не «проверять нечего»."""
    code, report = guard.evaluate(CASE_DIR, tmp_path / "nope.yaml")
    assert code == 1
    assert _classes(report) == {"map-unverifiable"}
    assert report["checked"] == 0


def test_defective_map_is_not_pass(tmp_path):
    """Своп на поле с drift: true — дефект карты: атомы пересеклись бы."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        r'''
version: 1
adr_glob: "docs/adr/ADR-009-*.md"
config: "net/config.json"
mappings:
  - id: a
    adr_pattern: "n_win = (\\d+)"
    config_field: "swa_window"
    drift: true
  - id: b
    adr_pattern: "головы, дом (\\d+)"
    config_field: "mla_index_dim"
    drift: true
swaps:
  - id: s
    between: ["a", "b"]
''',
        encoding="utf-8",
    )
    code, report = guard.evaluate(CASE_DIR, bad)
    assert code == 1
    assert _classes(report) == {"map-unverifiable"}
    assert any("drift: true" in f["message"] for f in report["findings"])


def test_map_loader_rejects_broken_yaml(tmp_path):
    """Загрузчик падает громко на своём подмножестве YAML, а не угадывает."""
    odd_indent = tmp_path / "odd.yaml"
    odd_indent.write_text("mappings:\n   - id: a\n", encoding="utf-8")
    bad_scalar = tmp_path / "bad-scalar.yaml"
    bad_scalar.write_text('config: "net\\x/config.json"\n', encoding="utf-8")

    with pytest.raises(acm.MapError):  # отступ вне поддержанных уровней
        acm.load(odd_indent)
    with pytest.raises(acm.MapError):  # значение не разбирается как JSON-скаляр
        acm.load(bad_scalar)
    with pytest.raises(acm.MapError):  # файла нет
        acm.load(tmp_path / "absent.yaml")


def test_case_map_overrides_script_map(tmp_path):
    """Карта кейса (карта едет со снапшотом) приоритетнее карты рядом со скриптом."""
    case = _mini_case(tmp_path)
    (case / "tools").mkdir()
    (case / "tools" / "adr_config_map.yaml").write_text(
        (TOOLS_DIR / "adr_config_map.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    code, report = guard.evaluate(case)
    assert code == 0
    assert report["map"] == (case / "tools" / "adr_config_map.yaml").as_posix()


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_exit_codes_and_json(tmp_path, capsys):
    """CLI: exit 0/1 и JSON-отчёт в stdout."""
    assert guard.main(["--case", str(CASE_DIR)]) == 0
    assert "PASS" in capsys.readouterr().out

    case = _mini_case(tmp_path)
    config = _config(case)
    config["num_layers"] = 48
    _write_config(case, config)

    assert guard.main(["--case", str(case), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["findings"][0]["class"] == "adr-config-mismatch"

    assert guard.main(["--case", str(case), "--quiet"]) == 1
    assert capsys.readouterr().out.startswith("FAIL C-047")


# ── связка с мутатором порчи (§9: порча → страж краснеет, откат → зелёный) ────

@pytest.fixture(scope="module")
def clean_case(tmp_path_factory) -> Path:
    """Полный чистый кейс: мутатору нужны model/, docs/, tools/ целиком."""
    dst = tmp_path_factory.mktemp("c047-case") / "case"
    copy_case_snapshot(CASE_DIR, dst)
    return dst


@pytest.mark.parametrize("kind", C047_ATOMS)
def test_mutator_drift_is_caught_and_reverted(clean_case, tmp_path, kind):
    """Интеграция: смысловая порча ADR/config → exit 1 с находкой; откат → exit 0."""
    ws = tmp_path / f"ws-{kind}"
    copy_case_snapshot(clean_case, ws)
    damage = next(
        d for d in corruption.plan_damages(clean_case, 3, "L3") if d.kind == kind
    )
    corruption.apply_damage(ws, damage)

    code, report = guard.evaluate(ws)
    assert code == 1, f"{kind}: связка «порча → страж» разорвана: {report}"
    assert _classes(report) == {"adr-config-mismatch"}
    field = damage.meta["field"] if "field" in damage.meta else damage.meta["fields"][0]["field"]
    assert field in {f["config_field"] for f in report["findings"]}

    corruption.revert_damage(ws, clean_case, damage)
    code_after, report_after = guard.evaluate(ws)
    assert code_after == 0, f"{kind}: откат не снял находку: {report_after}"


def test_claim_inversion_is_not_caught_by_guard(clean_case, tmp_path):
    """Граница стража: инверсия утверждения — класс модель-детекции, не C-047.

    Тест фиксирует, что связка «порча → C-047» НЕ распространяется на
    ``claim_inversion``: страж молчит, числа не тронуты. Так заявленная в §9
    специализация стражей остаётся проверяемым фактом, а не пожеланием.
    """
    ws = tmp_path / "ws-claim"
    copy_case_snapshot(clean_case, ws)
    damage = next(
        d for d in corruption.plan_damages(clean_case, 3, "L3") if d.kind == "claim_inversion"
    )
    corruption.apply_damage(ws, damage)

    code, report = guard.evaluate(ws)
    assert code == 0, f"claim_inversion не входит в предмет C-047: {report['findings']}"
    assert damage.meta["new"] in (ws / damage.file).read_text(encoding="utf-8")
