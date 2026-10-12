# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native GPU identity receipts for the GLM-5.3-Flash vLLM FPM producer."""


def native_gpu_identity(torch_module) -> dict:
    """Read this worker's selected CUDA device, without inference from flags."""
    index = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(index)
    identity = {
        "schema": "glm53flash_gpu_identity_v1",
        "name": torch_module.cuda.get_device_name(index),
        "compute_capability": list(torch_module.cuda.get_device_capability(index)),
        "total_memory_bytes": properties.total_memory,
        "cuda_device_index": index,
    }
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        identity["uuid"] = str(uuid)
    return identity


def validate_gb300_identity(identity: dict) -> None:
    """Admit actual GB300/sm103 receipts; caller separately binds TP rank."""
    import re

    if not isinstance(identity, dict) or identity.get("schema") != "glm53flash_gpu_identity_v1":
        raise ValueError("native GPU identity receipt is missing or unknown")
    name = identity.get("name")
    capability = identity.get("compute_capability")
    if (
        not isinstance(name, str)
        or re.search(r"\bGB300\b", name, re.IGNORECASE) is None
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
        or capability != [10, 3]
        or type(identity.get("total_memory_bytes")) is not int
        or identity["total_memory_bytes"] <= 0
        or type(identity.get("cuda_device_index")) is not int
        or identity["cuda_device_index"] < 0
        or ("uuid" in identity and (not isinstance(identity["uuid"], str) or not identity["uuid"].strip()))
    ):
        raise ValueError("native GPU identity does not attest GB300 with sm103 and valid device properties")
