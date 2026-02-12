from __future__ import annotations

import subprocess
import sys

MODULES = [
    "tuned_models.tune_classical_models",
    "tuned_models.tune_bilstm",
    "tuned_models.tune_distilbert",
    "tuned_models.tune_textgcn",
]


def main() -> None:
    for mod in MODULES:
        cmd = [sys.executable, "-m", mod]
        print("\n>>", " ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
