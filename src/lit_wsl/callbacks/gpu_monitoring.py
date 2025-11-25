import os
import shutil
import subprocess  # nosec
import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.accelerators.cuda import CUDAAccelerator
from lightning.pytorch.utilities import rank_zero_deprecation, rank_zero_only
from lightning.pytorch.utilities.exceptions import (
    MisconfigurationException,  # pyright: ignore[reportPrivateImportUsage]
)
from lightning.pytorch.utilities.parsing import AttributeDict
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch


class NvidiaGPUStatsMonitor(pl.Callback):
    r"""Now deprecated GPU only monitor.

    Automatically monitors and logs GPU stats during training stage. ``NvidiaGPUStatsMonitor``
    is a callback and in order to use it you need to assign a logger in the ``Trainer``.

    Args:
        memory_utilization: Set to ``True`` to monitor used, free and percentage of memory
            utilization at the start and end of each step. Default: ``True``.
        gpu_utilization: Set to ``True`` to monitor percentage of GPU utilization
            at the start and end of each step. Default: ``True``.
        intra_step_time: Set to ``True`` to monitor the time of each step. Default: ``False``.
        inter_step_time: Set to ``True`` to monitor the time between the end of one step
            and the start of the next step. Default: ``False``.
        fan_speed: Set to ``True`` to monitor percentage of fan speed. Default: ``False``.
        temperature: Set to ``True`` to monitor the memory and gpu temperature in degree Celsius.
            Default: ``False``.

    Raises:
        MisconfigurationException:
            If NVIDIA driver is not installed, not running on GPUs, or ``Trainer`` has no logger.

    Example::

        >>> import lightning.pytorch as pl
        >>> from lit_wsl.callbacks import GPUStatsMonitor
        >>> gpu_stats = GPUStatsMonitor() # doctest: +SKIP
        >>> trainer = plTrainer(callbacks=[gpu_stats]) # doctest: +SKIP

    GPU stats are mainly based on `nvidia-smi --query-gpu` command. The description of the queries is as follows:

    - **fan.speed** – The fan speed value is the percent of maximum speed that the device's fan is currently
      intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
      If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
      Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
    - **memory.used** – Total memory allocated by active contexts.
    - **memory.free** – Total free memory.
    - **utilization.gpu** – Percent of time over the past sample period during which one or more kernels was
      executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
    - **utilization.memory** – Percent of time over the past sample period during which global (device) memory was
      being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
    - **temperature.gpu** – Core GPU temperature, in degrees C.
    - **temperature.memory** – HBM memory temperature, in degrees C.

    """

    def __init__(
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ):
        super().__init__()

        if shutil.which("nvidia-smi") is None:
            raise MisconfigurationException(
                "Cannot use NvidiaGPUStatsMonitor callback because NVIDIA driver is not installed."
            )

        self._log_stats = AttributeDict(
            {
                "memory_utilization": memory_utilization,
                "gpu_utilization": gpu_utilization,
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "fan_speed": fan_speed,
                "temperature": temperature,
            }
        )

        # The logical device IDs for selected devices
        self._device_ids: list[int] = []  # will be assigned later in setup()

        # The unmasked real GPU IDs
        self._gpu_ids: list[str] = []  # will be assigned later in setup()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str | None = None) -> None:
        if not trainer.logger:
            raise MisconfigurationException("Cannot use GPUStatsMonitor callback with Trainer that has no logger.")

        if not CUDAAccelerator.is_available():
            raise MisconfigurationException("You are using GPUStatsMonitor but the CUDA accelerator is not available. ")

        # The logical device IDs for selected devices
        # ignoring mypy check because `trainer.data_parallel_device_ids` is None when using CPU
        self._device_ids = sorted(set(trainer.device_ids))  # type: ignore[reportGeneralTypeIssues]

        # The unmasked real GPU IDs
        self._gpu_ids = self._get_gpu_ids(self._device_ids)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_intra_step_time: float | None = None
        self._snap_inter_step_time: float | None = None

    @rank_zero_only
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = self._get_gpu_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._device_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["batch_time/inter_step (ms)"] = (time.time() - self._snap_inter_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._device_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["batch_time/intra_step (ms)"] = (time.time() - self._snap_intra_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @staticmethod
    def _get_gpu_ids(device_ids: list[int]) -> list[str]:
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: list[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return [cuda_visible_devices[device_id].strip() for device_id in device_ids]

    def _get_gpu_stats(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []

        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ",".join(queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self._gpu_ids)
        result: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603 # nosec
            [
                # it's ok to supress the warning here since we ensure nvidia-smi exists during init
                shutil.which("nvidia-smi"),  # type: ignore  # noqa: PGH003
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={gpu_ids}",
            ],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        stats = [[_to_float(x) for x in s.split(", ")] for s in result.stdout.strip().split(os.linesep)]
        return stats

    @staticmethod
    def _parse_gpu_stats(
        device_ids: list[int], stats: list[list[float]], keys: list[tuple[str, str]]
    ) -> dict[str, float]:
        """Parse the gpu stats into a loggable dict."""
        logs = {}
        for i, device_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                logs[f"device_id: {device_id}/{x} ({unit})"] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> list[tuple[str, str]]:
        """Get the GPU stats keys."""
        stat_keys = []

        if self._log_stats.gpu_utilization:
            stat_keys.append(("utilization.gpu", "%"))

        if self._log_stats.memory_utilization:
            stat_keys.extend([("memory.used", "MB"), ("memory.free", "MB"), ("utilization.memory", "%")])

        return stat_keys

    def _get_gpu_device_stat_keys(self) -> list[tuple[str, str]]:
        """Get the device stats keys."""
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(("fan.speed", "%"))

        if self._log_stats.temperature:
            stat_keys.extend([("temperature.gpu", "°C"), ("temperature.memory", "°C")])

        return stat_keys
