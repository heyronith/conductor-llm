"""Live progress reporter with Chicago/UTC timestamps, 1..100% integer steps, and cost tracking."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

from ccpt.training.cost import GPU_HOURLY_PRICES

GPU_PRICES = GPU_HOURLY_PRICES


class LiveProgressReporter:

    """Emits unbuffered real-time progress logs from 1/100 to 100/100 and persists records to JSONL."""

    def __init__(
        self,
        task_name: str,
        total_steps: int,
        total_tokens: int,
        model_name: Optional[str] = None,
        phase: str = "LM",
        gpu_type: str = "H100!",
        jsonl_path: Optional[Union[str, Path]] = None,
        require_jsonl: bool = False,
    ):
        if require_jsonl and not jsonl_path:
            raise ValueError("require_jsonl=True but jsonl_path is None or empty. Production reporting requires explicit JSONL persistence.")

        self.task_name = task_name
        self.total_steps = total_steps
        self.total_tokens = total_tokens
        self.model_name = model_name
        self.phase = phase
        self.gpu_type = gpu_type
        self.require_jsonl = require_jsonl
        self.gpu_price_per_sec = GPU_PRICES.get(gpu_type, 3.9492) / 3600.0

        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.chicago_tz = ZoneInfo("America/Chicago")
        self.start_time = time.time()
        self.last_reported_pct = 0
        self.loss_ema = None
        self.ema_alpha = 0.05

    def update_loss_ema(self, current_loss: float) -> float:
        """Updates exponential moving average of loss."""
        if not math.isfinite(current_loss):
            return current_loss
        if self.loss_ema is None:
            self.loss_ema = current_loss
        else:
            self.loss_ema = (1.0 - self.ema_alpha) * self.loss_ema + self.ema_alpha * current_loss
        return self.loss_ema

    def format_time(self, seconds: float) -> str:
        """Formats seconds into HH:MM:SS."""
        if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
            return "??:??:??"
        total_sec = int(seconds)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def step(
        self,
        current_step: int,
        tokens_seen: int,
        current_loss: Optional[float] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        token_acc: Optional[float] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Checks if a new integer percentage threshold has been reached and logs it."""
        record = None
        pct = min(100, max(1, int(math.floor(100.0 * float(current_step) / float(self.total_steps)))))

        if not force and pct <= self.last_reported_pct:
            return None

        # Emit all missed percentages up to current pct to ensure continuous 1..100 coverage
        target_pct = pct
        while self.last_reported_pct < target_pct:
            self.last_reported_pct += 1
            reported_pct = self.last_reported_pct

            now = time.time()
            elapsed_sec = max(0.001, now - self.start_time)
            progress_frac = float(reported_pct) / 100.0

            # Calculate ETA
            rate = float(current_step) / elapsed_sec if current_step > 0 else 0.0
            remaining_steps = max(0, self.total_steps - current_step)
            eta_sec = remaining_steps / rate if rate > 0.0 else 0.0

            # Tokens / sec
            tok_per_sec = float(tokens_seen) / elapsed_sec if tokens_seen > 0 else 0.0

            # Cost accounting
            cost_so_far = elapsed_sec * self.gpu_price_per_sec
            projected_cost = (elapsed_sec + eta_sec) * self.gpu_price_per_sec

            # Loss EMA
            ema = self.update_loss_ema(current_loss) if current_loss is not None else None

            # Timestamps
            now_dt = datetime.now(timezone.utc)
            chicago_str = now_dt.astimezone(self.chicago_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
            utc_str = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Construct log entry
            model_tag = f"[MODEL={self.model_name}]" if self.model_name else ""
            log_line = (
                f"[{chicago_str} | {utc_str}]"
                f"[{self.task_name}]{model_tag}[PHASE={self.phase}] "
                f"PROGRESS={reported_pct}/100 "
                f"elapsed={self.format_time(elapsed_sec)} "
                f"eta={self.format_time(eta_sec)} "
                f"step={current_step}/{self.total_steps} "
                f"tokens={tokens_seen:,}/{self.total_tokens:,} "
            )

            if current_loss is not None:
                log_line += f"loss={current_loss:.4f} loss_ema={ema:.4f} "
            if token_acc is not None:
                log_line += f"acc={token_acc*100:.1f}% "
            if lr is not None:
                log_line += f"lr={lr:.2e} "
            if grad_norm is not None:
                log_line += f"grad_norm={grad_norm:.2f} "

            log_line += (
                f"tok_s={tok_per_sec:,.0f} "
                f"gpu={self.gpu_type} "
                f"cost_so_far=${cost_so_far:.2f} "
                f"projected_cost=${projected_cost:.2f}"
            )

            if extra_info:
                for k, v in extra_info.items():
                    if isinstance(v, float):
                        log_line += f" {k}={v:.4f}"
                    else:
                        log_line += f" {k}={v}"

            # Flush immediately to stdout
            print(log_line, flush=True)

            # VRAM tracking if torch.cuda available
            vram_allocated_gb = 0.0
            vram_reserved_gb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    vram_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    vram_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            except Exception:
                pass

            record = {
                "chicago_time": chicago_str,
                "utc_time": utc_str,
                "timestamp_epoch": now,
                "task": self.task_name,
                "model": self.model_name,
                "phase": self.phase,
                "progress_pct": reported_pct,
                "elapsed_seconds": elapsed_sec,
                "measured_elapsed_gpu_seconds": elapsed_sec,
                "eta_seconds": eta_sec,
                "current_step": current_step,
                "total_steps": self.total_steps,
                "tokens_seen": tokens_seen,
                "total_tokens": self.total_tokens,
                "loss": current_loss,
                "loss_ema": ema,
                "token_acc": token_acc,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": tok_per_sec,
                "gpu_type": self.gpu_type,
                "vram_allocated_gb": vram_allocated_gb,
                "vram_reserved_gb": vram_reserved_gb,
                "cost_so_far_usd": cost_so_far,
                "accrued_cost_usd": cost_so_far,
                "projected_cost_usd": projected_cost,
                "extra_info": extra_info or {},
            }

            if self.jsonl_path:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        return record
