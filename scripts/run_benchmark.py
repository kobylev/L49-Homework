import subprocess
import os
import pandas as pd

def main():
    print("Running Benchmarks...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(["python", "scripts/run_all_experiments.py"], env=env)
    
    if os.path.exists("output/results.csv"):
        df = pd.DataFrame(pd.read_csv("output/results.csv"))
        print("\n--- Benchmark Results ---")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
