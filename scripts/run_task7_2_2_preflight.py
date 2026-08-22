"""Task 7.2.2 Final Preflight Proofs Script.

Invokes Modal bounded proofs and synthesizes final preflight summary and documentation.
"""

import subprocess
import sys


def main():
    print("Launching Modal Task 7.2.2 Final Preflight Proofs...")
    cmd = ["modal", "run", "modal/task7_2_2_final_preflight.py"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Modal execution failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print("✓ Modal Task 7.2.2 Final Preflight completed successfully.")


if __name__ == "__main__":
    main()
