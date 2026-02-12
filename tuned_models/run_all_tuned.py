from __future__ import annotations

import subprocess
import sys


SCRIPTS = [
    "tuned_models/tune_classical_models.py",
    "tuned_models/tune_bilstm.py",
    "tuned_models/tune_distilbert.py",
    "tuned_models/tune_textgcn.py",
]


def main() -> None:
    for script in SCRIPTS:
        cmd = [sys.executable, script]
        print("\n>>", " ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
