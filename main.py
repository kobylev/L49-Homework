import argparse
from scripts.run_all_experiments import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-experiments", action="store_true", help="Run all 8 experiments")
    args = parser.parse_args()
    
    if args.all_experiments:
        run_all()
    else:
        print("Please use --all-experiments to run the project.")
