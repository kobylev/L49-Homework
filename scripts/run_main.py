import subprocess
import os

def main():
    print("Starting Main Next-Word Prediction Pipeline...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(["python", "scripts/run_all_experiments.py"], env=env)

if __name__ == "__main__":
    main()
