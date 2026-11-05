"""End-to-end smoke test: the main script runs and writes a Tecplot file."""
import os
import shutil
import subprocess
import sys

from fvm_helpers import REPO_ROOT


def test_main_script_runs_on_analytical_fields(tmp_path):
    for item in ["A_main.py", "InputTools.py", "CalcOperators", "GridTools", "Functions", "Grids"]:
        src = os.path.join(REPO_ROOT, item)
        dst = tmp_path / item
        shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy(src, dst)
    (tmp_path / "test_input.txt").write_text("mesh=Grids/base.msh\niterations=2\nmod=0\n")

    env = dict(os.environ, FVM_INPUT="test_input.txt")
    result = subprocess.run([sys.executable, "A_main.py"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert "Work done!" in result.stdout
    output = tmp_path / "Results" / "data0.plt"
    assert output.exists()
    assert output.read_text().startswith("VARIABLES")
