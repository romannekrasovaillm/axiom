"""S-003 replica_bytes — тесты датчика."""

from tools.sensors import replica_bytes
from tools.sensors._common import REPO_ROOT


def test_selftest_green():
    assert replica_bytes.run_selftest() == 0


def test_measure_additivity(tmp_path):
    written = replica_bytes.measure(REPO_ROOT / "net" / "config.json", out_dir=tmp_path)
    params = written["replica_bytes_params"]["value"]
    opt = written["replica_bytes_opt_state"]["value"]
    assert written["replica_bytes_total"]["value"] == params + opt
    assert params > 0 and opt > 0
