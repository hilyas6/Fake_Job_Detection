# run_all.py
import subprocess
import sys

def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    run([sys.executable, "-m", "src.prep"])
    run([sys.executable, "-m", "src.train_baselines"])
    run([sys.executable, "-m", "src.evaluate", "--model", "baselines"])
    # SHAP optional
    try:
        run([sys.executable, "-m", "src.explain", "--model", "tfidf_lr"])
    except Exception as e:
        print("SHAP step skipped/failed:", e)

if __name__ == "__main__":
    main()
