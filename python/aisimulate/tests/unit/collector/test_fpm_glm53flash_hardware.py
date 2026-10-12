# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
from collector.fpm_forward.runtime.glm53flash import glm53flash_worker_hardware as hardware

pytestmark = pytest.mark.unit


def native_worker(monkeypatch, tmp_path, *, name="NVIDIA GB300", rank=0):
    source = tmp_path / "gpu_worker.py"
    source.write_text("# test-only native worker source\n")
    output = tmp_path / "benchmark.json"
    monkeypatch.setenv("DYN_FPM_BENCHMARK_OUTPUT_PATH", str(output))
    (tmp_path / "collector-provenance.json").write_text('{"attempt_id":"test-attempt"}\n')
    monkeypatch.setattr(hardware.importlib.metadata, "version", lambda _: "0.31.0")
    ready = []

    class Worker:
        def init_device(self, value):
            ready.append(value)
            self.device = f"cuda:{rank}"
            return "native-return"

    def current_device():
        assert ready, "device properties must be read after native initialization"
        return rank

    cuda = SimpleNamespace(
        current_device=current_device,
        get_device_name=lambda _: name,
        get_device_capability=lambda _: (10, 3),
        get_device_properties=lambda _: SimpleNamespace(total_memory=288 * 1024**3, uuid=f"GPU-{rank}"),
    )
    module = SimpleNamespace(Worker=Worker, torch=SimpleNamespace(cuda=cuda), __file__=str(source))
    hardware.install(module)
    hardware.install(module)  # One callback, even when imported again.
    worker = Worker()
    worker.rank = rank
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1)
    return worker, ready


def test_native_initializer_return_and_selected_device_are_preserved(monkeypatch, tmp_path):
    worker, calls = native_worker(monkeypatch, tmp_path)
    assert worker.init_device("native-input") == "native-return"
    assert calls == ["native-input"]
    receipt = json.loads((tmp_path / "native-device-rank-0.json").read_text())
    assert receipt["hardware"]["uuid"] == "GPU-0"
    assert receipt["status"] == "passed"
    assert (
        receipt["collector_provenance_sha256"]
        == hashlib.sha256((tmp_path / "collector-provenance.json").read_bytes()).hexdigest()
    )
    assert not list(tmp_path.glob("benchmark*.json")), "device receipts are not benchmark envelopes"
    with pytest.raises(FileExistsError):
        worker.init_device("repeat-attempt")


def test_rejected_native_device_preserves_failure_receipt(monkeypatch, tmp_path):
    worker, _ = native_worker(monkeypatch, tmp_path, name="NVIDIA B300")
    with pytest.raises(ValueError, match="GB300"):
        worker.init_device("native-input")
    receipt = json.loads((tmp_path / "native-device-rank-0.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["hardware"]["name"] == "NVIDIA B300"
