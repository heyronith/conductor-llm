"""Configuration and default environment paths for CCPT data pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_if_present() -> None:
    """Safely populate environment variables from local .env if present."""
    env_file = Path(".env")
    if env_file.exists():
        try:
            content = env_file.read_text(encoding="utf-8").strip()
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
                elif line.startswith("hf_") and "HF_TOKEN" not in os.environ:
                    os.environ["HF_TOKEN"] = line
        except Exception:
            pass


load_dotenv_if_present()

DEFAULT_DATA_ROOT = os.environ.get("CCPT_DATA_ROOT", "data/processed")
DATA_ORDER_SEED = 20260820
FORMAT_VERSION = "ccpt-data-v1"


@dataclass(frozen=True)
class DataConfig:
    """Immutable data configuration for CCPT dataset and tokenization pipeline."""

    data_root: str = DEFAULT_DATA_ROOT
    tokenizer_repo: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_revision: str = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
    fineweb_repo: str = "HuggingFaceFW/fineweb-edu"
    fineweb_revision: str = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    fineweb_config: str = "sample-100BT"
    wildguard_repo: str = "allenai/wildguardmix"
    wildguard_revision: str = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
    wildguard_train_config: str = "wildguardtrain"
    wildguard_test_config: str = "wildguardtest"
    max_seq_len: int = 1024
    lm_target_tokens: int = 10_000_000_000
    safety_target_tokens: int = 100_000_000
    data_seed: int = DATA_ORDER_SEED
    format_version: str = FORMAT_VERSION

    def get_data_path(self, *subpaths: str) -> Path:
        """Resolve a subpath under the configured CCPT_DATA_ROOT."""
        return Path(self.data_root).joinpath(*subpaths)
