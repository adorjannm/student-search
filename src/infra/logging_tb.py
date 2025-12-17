from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from src.infra.config import RunContext

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


class Logger(Protocol):
    run: Optional[RunContext]

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None: ...

    def log_text(self, tag: str, text: str, step: int = 0) -> None: ...

    def log_hparams(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None: ...

    def flush(self) -> None: ...
    def close(self) -> None: ...


class NoOpLogger:
    run: Optional[RunContext] = None

    def log_scalar(self, *_: Any, **__: Any) -> None:
        return

    def log_dict(self, *_: Any, **__: Any) -> None:
        return

    def log_text(self, *_: Any, **__: Any) -> None:
        return

    def log_hparams(self, *_: Any, **__: Any) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class TensorboardLogger:
    def __init__(self, writer: Any, ctx: RunContext):
        self._w = writer
        self.run = ctx

    @classmethod
    def create(cls, ctx: RunContext, enabled: bool = True) -> Logger:
        if not enabled or SummaryWriter is None:
            return NoOpLogger()

        ctx.ensure_dirs()
        writer = SummaryWriter(log_dir=str(ctx.tb_run_dir))
        return cls(writer, ctx)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._w.add_scalar(tag, float(value), int(step))

    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        for k, v in values.items():
            if v is None:
                continue
            self.log_scalar(f"{prefix}/{k}", float(v), step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self._w.add_text(tag, text, int(step))

    def log_hparams(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None:
        # TensorBoard expects scalars in metrics; hparams can be any JSON-like.
        self._w.add_hparams(dict(hparams), {k: float(v) for k, v in metrics.items()})

    def flush(self) -> None:
        self._w.flush()

    def close(self) -> None:
        self._w.close()
