from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuned_models.common import MODELS_DIR

MODULES = [
    "tuned_models.tune_classical_models",
    "tuned_models.tune_bilstm",
    "tuned_models.tune_distilbert",
    "tuned_models.tune_textgcn",
]

MODEL_ARTIFACTS = {
    "tuned_models.tune_classical_models": [
        MODELS_DIR / "vectorizer.joblib",
        MODELS_DIR / "logistic_regression.joblib",
        MODELS_DIR / "naive_bayes.joblib",
        MODELS_DIR / "random_forest.joblib",
        MODELS_DIR / "xgboost.joblib",
        MODELS_DIR / "lightgbm.joblib",
    ],
    "tuned_models.tune_bilstm": [MODELS_DIR / "bilstm.pt"],
    "tuned_models.tune_distilbert": [MODELS_DIR / "distilbert"],
    "tuned_models.tune_textgcn": [MODELS_DIR / "textgcn.pt", MODELS_DIR / "textgcn_vectorizer.joblib"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all tuned model trainers with visible progress.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m.split(".")[-1] for m in MODULES],
        help="Optional subset to run (default: all). Example: --models tune_bilstm tune_textgcn",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a trainer if its expected output artifact(s) already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining trainers even if one fails.",
    )
    return parser.parse_args()


def artifacts_exist(module: str) -> bool:
    expected = MODEL_ARTIFACTS.get(module, [])
    return bool(expected) and all(Path(p).exists() for p in expected)


def main() -> None:
    args = parse_args()
    selected = set(args.models or [m.split(".")[-1] for m in MODULES])
    modules = [m for m in MODULES if m.split(".")[-1] in selected]

    print(f"Running {len(modules)} tuned trainer(s)...", flush=True)
    failures = []
    for i, mod in enumerate(modules, start=1):
        if args.skip_completed and artifacts_exist(mod):
            print(f"[{i}/{len(modules)}] Skipping {mod} (artifacts already exist)", flush=True)
            continue

        cmd = [sys.executable, "-u", "-m", mod]
        print(f"\n[{i}/{len(modules)}] Starting {mod}", flush=True)
        print(">>", " ".join(cmd), flush=True)
        start = time.perf_counter()

        env = dict(os.environ, PYTHONUNBUFFERED="1")
        result = subprocess.run(cmd, env=env)
        elapsed = time.perf_counter() - start

        if result.returncode == 0:
            print(f"[{i}/{len(modules)}] Finished {mod} in {elapsed:.1f}s", flush=True)
        else:
            print(f"[{i}/{len(modules)}] FAILED {mod} in {elapsed:.1f}s", flush=True)
            failures.append(mod)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        raise SystemExit(f"Failed modules: {', '.join(failures)}")
    print("All selected tuned trainers completed.", flush=True)


if __name__ == "__main__":
    main()
