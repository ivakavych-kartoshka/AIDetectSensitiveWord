import runpy
import sys

sys.path.insert(0, ".")

runpy.run_path(
    "src/evaluation/evaluate_per_model.py",
    run_name="__main__",
)