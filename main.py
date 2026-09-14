"""Main entry point for MCQ Solver."""
import sys
from src.stage1_baseline import run_baseline


def main():
    print("MCQ Solver - Stage 1 Baseline (TF-IDF)")
    print("-" * 50)
    run_baseline()


if __name__ == "__main__":
    main()