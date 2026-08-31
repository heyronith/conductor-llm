"""Modal CPU launcher to materialize the 128k FineWeb continuation on persistent volume."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

app = modal.App("strengthening-fineweb-materializer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        "sentencepiece==0.2.0",
        "tiktoken==0.8.0",
        "accelerate==1.1.1",
        "pyarrow==17.0.0",
        "numpy==2.1.3",
        "pytest==8.3.3",
    )
    .add_local_python_source("ccpt")
)

data_volume = modal.Volume.from_name("ccpt-authoritative-data", create_if_missing=True)
hf_secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=hf_secrets,
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def run_materialization(code_sha: str):
    from ccpt.data.strengthening_materializer import materialize_strengthening_fineweb_continuation

    print("=== Starting Modal CPU Materialization of Strengthening Continuation ===", flush=True)
    t0 = time.time()
    res = materialize_strengthening_fineweb_continuation(
        authoritative_dir="/data/fineweb_authoritative",
        output_dir="/data/fineweb_strengthening",
        code_sha=code_sha,
    )
    data_volume.commit()
    elapsed = time.time() - t0
    print(f"Materialization completed in {elapsed:.2f}s!", flush=True)
    return res


def main():
    res_git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    code_sha = res_git.stdout.strip()
    print(f"Executing materialization for code SHA: {code_sha}")

    with app.run():
        result = run_materialization.remote(code_sha=code_sha)
        print("Materialization result:")
        print(json.dumps(result, indent=2))

        # Save manifest locally to artifacts/strengthening_task2_extended_fineweb_manifest.json
        manifest = result.get("manifest", {})
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        manifest_path = artifacts_dir / "strengthening_task2_extended_fineweb_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
