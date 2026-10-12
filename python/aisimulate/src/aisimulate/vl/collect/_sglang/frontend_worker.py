# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record SGLang frontend service times for one image workload; prints one JSON document.

Runs under the serving host's interpreter that has sgl-project/sglang v0.5.19
installed (Python 3.10 or newer, standard library plus sglang). Both frontends
are measured as one request-level `process` black box under a closed loop of
one request up to the width of the pool being measured, on the real SGLang
objects:

* Python frontend: the tokenizer manager's multimodal processor is instantiated
  in this process with its own IO and processor pools; `process` is the wall
  time of `process_mm_data_async` (image load, HF processor, layout).
* Rust frontend: the embedded Rust server is launched alone (no GPU, no
  scheduler) on a single rank; `process` is the time from the HTTP send to the
  moment the scheduler-side drain returns the request. Each drained request is
  then answered with an error, which the client only uses to move on.

Not measured on either path: the tokenizer manager's serialization and
shared-memory publish of the features, the scheduler's per-request receive
preparation of the request, HTTP parsing and chat templating in the Python
server, and every scheduler cost beyond admission. Progress goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samples import MIN_STEADY_SAMPLES, Span, steady_samples
from workload import generate_images, image_data_url

SGLANG_VERSION = "0.5.19"
REQUEST_DEADLINE_S = 120.0


def _release(version: str) -> str:
    """`major.minor.patch` of a version string, local suffixes dropped (mirrors `vl.table.sglang_release`)."""
    return ".".join(version.split("+", 1)[0].split(".")[:3])


def _sglang_version() -> str:
    from sglang.version import __version__

    if _release(str(__version__)) != SGLANG_VERSION:
        raise SystemExit(f"installed sglang {__version__} is not {SGLANG_VERSION}; the stage boundaries are pinned")
    return str(__version__)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _serving_components(model, mm_process_config, port=None, image_processor_backend="pil"):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.tokenizer_manager import get_processor_wrapper
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer

    extra = {}
    if mm_process_config:
        extra["mm_process_config"] = mm_process_config
    if port is not None:
        extra.update(host="127.0.0.1", port=port)
    server_args = ServerArgs(
        model_path=model,
        dtype="bfloat16",
        enable_multimodal=True,
        skip_server_warmup=True,
        image_processor_backend=image_processor_backend,
        **extra,
    )
    set_global_server_args_for_tokenizer(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return server_args, model_config, get_processor_wrapper()


def _prompt(processor, count, text_tokens):
    tokenizer = processor.tokenizer
    ids = tokenizer.encode("Describe the images carefully. " * (text_tokens + 1))[:text_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"}] * count}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def closed_loop(call, level, budget_per_worker=100):
    """Keep `level` jobs in flight until the level has enough steady samples.

    A burst of short jobs never runs at its target concurrency: submission and
    wake-up skew exceed a few-millisecond service, so the burst drains before its
    last job starts. Resubmitting on completion keeps the load constant; ramp-up
    and drain samples fall out of the steady criterion.
    """
    spans = []
    state = {"issued": 0, "done": 0, "stop": False}
    budget = budget_per_worker * level

    async def worker():
        while not state["stop"] and state["issued"] < budget:
            index = state["issued"]
            state["issued"] += 1
            spans.append(await call(index))
            state["done"] += 1
            if state["done"] % level == 0 and len(steady_samples(spans, level)) >= MIN_STEADY_SAMPLES:
                state["stop"] = True

    await asyncio.gather(*(worker() for _ in range(level)))
    _progress(f"level {level}: {len(steady_samples(spans, level))} steady of {len(spans)} samples")
    return spans


async def measure_levels(call, levels):
    recorded = {}
    for level in range(1, levels + 1):
        spans = await closed_loop(call, level)
        recorded[str(level)] = [[span.started_ns, span.ended_ns] for span in spans]
    return recorded


def _feature_devices(items):
    devices = set()
    for item in items:
        features = item.feature if isinstance(item.feature, list) else [item.feature]
        devices.update(str(getattr(feature, "device", "cpu")) for feature in features if feature is not None)
    return devices


def _environment(version):
    """What stays fixed across collections on this host; the table's identity."""
    return {"cpu": _cpu_model(), "sglang_version": version, "python": sys.version.split()[0]}


async def measure_python(args, images, mm_process_config):
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode

    server_args, model_config, processor = _serving_components(args.model, mm_process_config)
    import_processors("sglang.srt.multimodal.processors")
    mm = get_mm_processor(
        model_config.hf_config, server_args, processor, determine_tensor_transport_mode(), model_config=model_config
    )
    urls = [image_data_url(encoded, args.encoding) for encoded in images]
    prompt = _prompt(processor, len(images), args.text_tokens)
    workers = int(mm.io_executor._max_workers)
    levels = workers
    devices = set()
    try:

        async def one(index):
            started = time.perf_counter_ns()
            output = await mm.process_mm_data_async(
                image_data=urls,
                input_text=prompt,
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid=f"collect-{index}"),
            )
            span = Span(started, time.perf_counter_ns())
            devices.update(_feature_devices(output.mm_items))
            one.output = output
            return span

        # Warm the pools and the processor before timing.
        await asyncio.gather(*(one(-1 - index) for index in range(2 * levels)))
        if devices != {"cpu"}:
            raise SystemExit(
                f"the Python frontend produced features on {sorted(devices)}; only the CPU path is modeled"
            )
        recorded = await measure_levels(one, levels)
        output = one.output
    finally:
        mm.shutdown()
    return {
        "frontend": "python",
        "workers": workers,
        "levels": recorded,
        "provenance": {
            "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
            "mm_processor_class": f"{type(mm).__module__}.{type(mm).__name__}",
            "image_processor_backend": mm.image_processor_backend,
            "processor_workers": mm.mm_processor_worker_num,
            "text_tokens": args.text_tokens,
            "input_tokens": len(output.input_ids),
            "includes": {
                "process": ["BaseMultimodalProcessor.process_mm_data_async (image load, HF processor, layout)"],
            },
        },
    }


async def measure_rust(args, images, mm_process_config):
    import aiohttp
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.rust_server.multimodal import RustMmProcessor
    from sglang.srt.rust_server.server import RustServer

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_args, model_config, processor = _serving_components(args.model, mm_process_config, port=port)
    workers = int(server_args.mm_processor_worker_num or getattr(RustMmProcessor, "AUTO_MM_WORKERS", 8))
    levels = workers
    descriptor = SimpleNamespace(
        server_args=server_args,
        model_config=model_config,
        processor=processor,
        ps=SimpleNamespace(attn_dp_rank=0, dp_size=1),
        max_total_num_tokens=model_config.context_len,
    )
    prompt = _prompt(processor, len(images), args.text_tokens)
    payload_images = [image_data_url(encoded, args.encoding) for encoded in images]
    done = {}
    stop = threading.Event()
    failure = []

    def drain(server):
        # The scheduler-side drain marks the worker's completion. Control messages
        # (client aborts after our error reply) carry no work.
        try:
            while not stop.is_set():
                items = server.drain(256)
                now = time.perf_counter_ns()
                for item in items:
                    if not isinstance(item, TokenizedGenerateReqInput):
                        continue
                    done[str(item.rid).split("#", 1)[0]] = now
                    server.server.push_error(item.rid, "aisimulate collect: frontend-only recording")
                if not items:
                    time.sleep(0.0002)
        except Exception as exc:
            failure.append(exc)

    server = RustServer.launch(descriptor)
    if server.mm_spec is None:
        raise SystemExit("the Rust server did not enable its multimodal pipeline for this model")
    if server.mm_spec.feature_shm:
        raise SystemExit(
            "the Rust server chose shared-memory features; the collector models the inline single-rank path"
        )
    drainer = threading.Thread(target=drain, args=(server,), daemon=True)
    drainer.start()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_DEADLINE_S)) as session:

            async def one(index):
                rid = f"collect-{index}"
                started = time.perf_counter_ns()
                deadline = time.monotonic() + REQUEST_DEADLINE_S
                async with session.post(
                    f"http://127.0.0.1:{port}/generate",
                    json={
                        "rid": rid,
                        "text": prompt,
                        "image_data": payload_images,
                        "sampling_params": {"max_new_tokens": 1},
                    },
                ) as response:
                    body = await response.read()
                if rid not in done and response.status != 200:
                    raise RuntimeError(f"the Rust server rejected request {rid}: HTTP {response.status} {body[:200]!r}")
                while rid not in done:
                    if failure:
                        raise RuntimeError("the scheduler-side drain failed") from failure[0]
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"request {rid} never reached the scheduler-side drain")
                    await asyncio.sleep(0.0002)
                return Span(started, done.pop(rid))

            await asyncio.gather(*(one(-1 - index) for index in range(2 * levels)))
            recorded = await measure_levels(one, levels)
    finally:
        stop.set()
        drainer.join(timeout=5)
        server.server.shutdown()
    extension = sys.modules.get("sglang.srt.rust_extensions._server")
    return {
        "frontend": "rust",
        "workers": workers,
        "levels": recorded,
        "provenance": {
            "extension": getattr(extension, "__file__", None),
            "text_tokens": args.text_tokens,
            "includes": {
                "process": [
                    "HTTP receive and tokenization in the Rust server",
                    "multimodal worker: payload, fetch, hash, decode, patchify, layout, pack",
                    "channel to the scheduler drain",
                ],
            },
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="frontend_worker", description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frontend", choices=("python", "rust"), required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--encoding", choices=("png", "jpeg"), default="png")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    args = parser.parse_args(argv)
    version = _sglang_version()
    # Recorded before the Rust server narrows this thread's affinity.
    environment = _environment(version)
    collected = {"host": platform.node(), "threads": _threads()}
    bounds = (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels))
    image = {name: int(value) for name, value in bounds if value}
    mm_process_config = {"image": image} if image else None
    images = generate_images(args.height, args.width, args.count, args.encoding)
    measure = measure_python if args.frontend == "python" else measure_rust
    result = asyncio.run(measure(args, images, mm_process_config))
    result["environment"] = environment
    result["provenance"].update(collected, sampled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
