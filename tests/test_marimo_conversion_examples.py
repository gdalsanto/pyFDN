"""Test for marimo conversion."""

import os
import subprocess
import sys
from pathlib import Path

BASE = Path(
    __file__
).parent.parent.parent  # goes up from pyFDN/tests directory to the local root

directories = [
    BASE / "pyFDN" / "examples",  # pyf/examples/*.py
    BASE / "pyFDN" / "examples" / "allPass_FDN",  # pyf/examples/all/*.py
]
python_exe = sys.executable

failed = []

# Tell matplotlib to use a non-interactive backend to avoid plots while testing
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# suppress plotly browser tabs
env["PLOTLY_RENDERER"] = "json"


for directory in directories:
    if not directory.exists():
        print(f"WARNING: Directory not found, skipping: {directory}")
        continue

    for py_file in directory.glob("*.py"):
        print(f"Running: {py_file}")

        result = subprocess.run(
            [python_exe, str(py_file)],
            capture_output=True,
            text=True,
            env=env,
        )

    if result.returncode != 0:
        failed.append(py_file)
        print("FAILED")
        print(result.stderr)
    else:
        print("OK")

print("\nSummary")
print("-------------------")

if failed:
    print("Failed files:")
    for f in failed:
        print(f)
else:
    print("All files executed successfully.")
