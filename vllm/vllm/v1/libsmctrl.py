# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import threading
import time
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

LIBSMCTRL_ENV = "LIBSMCTRL_PATH"
LIBSMCTRL_DEFAULT = "libsmctrl.so"
LIBSMCTRL_DEFAULT_SUBDIR = ".libs"
VLLM_STREAM_MASK_ENV = "VLLM_STREAM_MASK"
VLLM_STREAM_MASK_TOUCH_LOG_ENV = "VLLM_STREAM_MASK_TOUCH_LOG"

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0


class _CUexecAffinitySmCount(ctypes.Structure):
    _fields_ = [("val", ctypes.c_uint)]


class _CUexecAffinityParamUnion(ctypes.Union):
    _fields_ = [("smCount", _CUexecAffinitySmCount)]


class _CUexecAffinityParam(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("type", ctypes.c_int),
        ("param", _CUexecAffinityParamUnion),
    ]


_lib: ctypes.CDLL | None = None
_lib_load_attempted = False
_sm_limit_ctx: ctypes.c_void_p | None = None
_libcuda: ctypes.CDLL | None = None
_ctx_lock = threading.Lock()
_mask_lock = threading.Lock()
_masked_stream_masks: dict[int, int] = {}
_cached_stream_mask: int | None = None
_stream_mask_loaded = False
_stream_mask_touch_count = 0
_last_applied_mask: int | None = None


def _append_stream_mask_touch_log(ts_ns: int) -> None:
    log_path = os.environ.get(VLLM_STREAM_MASK_TOUCH_LOG_ENV, "").strip()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(f"{int(ts_ns)}\n")
    except Exception:
        return


def _get_libsmctrl_path() -> Path:
    env_path = os.environ.get(LIBSMCTRL_ENV)
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / LIBSMCTRL_DEFAULT_SUBDIR / LIBSMCTRL_DEFAULT


def _load_libsmctrl() -> ctypes.CDLL | None:
    global _lib, _lib_load_attempted
    if _lib_load_attempted:
        return _lib

    _lib_load_attempted = True
    lib_path = _get_libsmctrl_path()
    try:
        if lib_path.exists():
            _lib = ctypes.CDLL(str(lib_path))
        else:
            _lib = ctypes.CDLL(LIBSMCTRL_DEFAULT)
        logger.info("libsmctrl loaded from %s", lib_path)
    except OSError as exc:
        logger.warning("Could not load libsmctrl.so: %s", exc)
        _lib = None

    return _lib


def _load_libcuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is not None:
        return _libcuda
    try:
        _libcuda = ctypes.CDLL("libcuda.so.1")
        return _libcuda
    except OSError as exc:
        raise RuntimeError("Unable to load libcuda.so.1") from exc


def _check_cu(result: int, call: str) -> None:
    if result != 0:
        name = "UNKNOWN"
        message = ""
        try:
            cuda = _load_libcuda()
            cu_get_name = cuda.cuGetErrorName
            cu_get_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu_get_name.restype = ctypes.c_int
            cu_get_str = cuda.cuGetErrorString
            cu_get_str.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu_get_str.restype = ctypes.c_int

            name_ptr = ctypes.c_char_p()
            if cu_get_name(result, ctypes.byref(name_ptr)) == 0 and name_ptr.value:
                name = name_ptr.value.decode("utf-8", errors="replace")

            message_ptr = ctypes.c_char_p()
            if cu_get_str(result, ctypes.byref(message_ptr)) == 0 and message_ptr.value:
                message = message_ptr.value.decode("utf-8", errors="replace")
        except Exception:
            pass

        raise RuntimeError(
            f"CUDA Driver API call failed: {call} returned {result} ({name}) {message}".strip()
        )


def create_sm_limited_context(active_sms: int, device_index: int = 0) -> None:
    global _sm_limit_ctx
    if active_sms < 1:
        raise ValueError("active_sms must be at least 1")

    with _ctx_lock:
        if _sm_limit_ctx is not None:
            cuda = _load_libcuda()
            cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
            cuda.cuCtxDestroy_v2.restype = ctypes.c_int
            result = cuda.cuCtxDestroy_v2(_sm_limit_ctx)
            _sm_limit_ctx = None
            _check_cu(result, "cuCtxDestroy_v2")

        cuda = _load_libcuda()
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        _check_cu(cuda.cuInit(0), "cuInit")

        if hasattr(cuda, "cuCtxGetCurrent"):
            cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            cuda.cuCtxGetCurrent.restype = ctypes.c_int
            existing_ctx = ctypes.c_void_p()
            _check_cu(cuda.cuCtxGetCurrent(ctypes.byref(existing_ctx)), "cuCtxGetCurrent")
            if existing_ctx.value:
                raise RuntimeError(
                    "A CUDA context is already current in this process. "
                    "SM-limited context must be created before any CUDA runtime initialization."
                )

        cu_dev = ctypes.c_int()
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGet.restype = ctypes.c_int
        _check_cu(cuda.cuDeviceGet(ctypes.byref(cu_dev), int(device_index)), "cuDeviceGet")

        total_sms = ctypes.c_int()
        cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetAttribute.restype = ctypes.c_int
        _check_cu(
            cuda.cuDeviceGetAttribute(
                ctypes.byref(total_sms),
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                cu_dev,
            ),
            "cuDeviceGetAttribute",
        )
        if active_sms > int(total_sms.value):
            raise ValueError(
                f"Requested active_sms={active_sms} exceeds device SM count={total_sms.value}"
            )

        affinity = _CUexecAffinityParam()
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT
        affinity.param.smCount.val = int(active_sms)

        ctx = ctypes.c_void_p()
        if not hasattr(cuda, "cuCtxCreate_v3"):
            raise RuntimeError("CUDA Driver API symbol cuCtxCreate_v3 is unavailable in this runtime.")
        cuda.cuCtxCreate_v3.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_CUexecAffinityParam),
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
        ]
        cuda.cuCtxCreate_v3.restype = ctypes.c_int
        _check_cu(
            cuda.cuCtxCreate_v3(ctypes.byref(ctx), ctypes.byref(affinity), 1, 0, cu_dev),
            "cuCtxCreate_v3",
        )

        if not hasattr(cuda, "cuCtxSetCurrent"):
            raise RuntimeError("CUDA Driver API symbol cuCtxSetCurrent is unavailable in this runtime.")
        cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cuda.cuCtxSetCurrent.restype = ctypes.c_int
        _check_cu(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

        _sm_limit_ctx = ctx


def destroy_sm_limited_context() -> None:
    global _sm_limit_ctx
    with _ctx_lock:
        if _sm_limit_ctx is None:
            return
        cuda = _load_libcuda()
        cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        cuda.cuCtxDestroy_v2.restype = ctypes.c_int
        result = cuda.cuCtxDestroy_v2(_sm_limit_ctx)
        _sm_limit_ctx = None
        _check_cu(result, "cuCtxDestroy_v2")


def set_stream_mask(stream_ptr: int, mask: int) -> None:
    if int(stream_ptr) == 0:
        return

    lib = _load_libsmctrl()
    if lib is None:
        raise RuntimeError("libsmctrl.so could not be loaded")
    func = getattr(lib, "libsmctrl_set_stream_mask", None)
    if func is None:
        func = getattr(lib, "set_stream_mask", None)
    if func is None:
        raise RuntimeError(
            "libsmctrl stream mask symbol not found; expected libsmctrl_set_stream_mask or set_stream_mask"
        )

    func.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    func.restype = None
    func(ctypes.c_void_p(stream_ptr), ctypes.c_uint64(mask))


def set_global_mask(mask: int) -> None:
    lib = _load_libsmctrl()
    if lib is None:
        raise RuntimeError("libsmctrl.so could not be loaded")
    func = getattr(lib, "libsmctrl_set_global_mask", None)
    if func is None:
        raise RuntimeError(
            "libsmctrl_set_global_mask not found in libsmctrl.so; upgrade to a version that supports the global mask API."
        )
    func.restype = None
    func.argtypes = [ctypes.c_uint64]
    func(ctypes.c_uint64(mask))


def build_sm_mask(total_tpcs: int, disabled_tpcs: list[int]) -> int:
    if total_tpcs < 1:
        raise ValueError("total_tpcs must be at least 1")
    mask = 0
    for tpc in disabled_tpcs:
        if tpc < 0 or tpc >= total_tpcs:
            raise ValueError(f"Disabled TPC index {tpc} is out of range for {total_tpcs} total TPCs")
        mask |= 1 << tpc
    return mask


def split_sm_masks(total_tpcs: int) -> tuple[int, int]:
    if total_tpcs < 2:
        raise ValueError("Cannot split SM masks for fewer than 2 TPCs. Use a single stream instead.")
    split = total_tpcs // 2
    mask_mm = build_sm_mask(total_tpcs, list(range(split)))
    mask_llm = build_sm_mask(total_tpcs, list(range(split, total_tpcs)))
    return mask_mm, mask_llm


def _parse_int_env(value: str) -> int:
    stripped = value.strip().lower()
    if stripped.startswith("0x"):
        return int(stripped, 16)
    return int(stripped)


def get_configured_stream_mask() -> int | None:
    global _cached_stream_mask, _stream_mask_loaded
    with _mask_lock:
        if _stream_mask_loaded:
            return _cached_stream_mask

        raw = os.environ.get(VLLM_STREAM_MASK_ENV, "").strip()
        if not raw:
            _cached_stream_mask = None
            _stream_mask_loaded = True
            return None

        mask = _parse_int_env(raw)
        if mask < 0:
            raise ValueError(f"{VLLM_STREAM_MASK_ENV} must be >= 0")
        _cached_stream_mask = mask
        _stream_mask_loaded = True
        return _cached_stream_mask


def reset_stream_mask_touch_count() -> None:
    global _stream_mask_touch_count
    with _mask_lock:
        _stream_mask_touch_count = 0


def get_stream_mask_touch_count() -> int:
    with _mask_lock:
        return int(_stream_mask_touch_count)


def count_stream_mask_touches_in_window(
    req_start_ts: float,
    req_end_ts: float,
    log_path: str | None = None,
) -> int:
    path_str = (log_path or os.environ.get(VLLM_STREAM_MASK_TOUCH_LOG_ENV, "")).strip()
    if not path_str:
        return 0
    path = Path(path_str)
    if not path.exists() or req_end_ts <= req_start_ts:
        return 0

    req_start_ns = int(req_start_ts * 1e9)
    req_end_ns = int(req_end_ts * 1e9)
    count = 0
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts_ns = int(line)
                except ValueError:
                    continue
                if req_start_ns <= ts_ns <= req_end_ns:
                    count += 1
    except Exception:
        return 0

    return count


def apply_configured_global_mask() -> bool:
    # Global-mask application is intentionally disabled to avoid mixing global
    # and stream-level masking paths during validation runs.
    logger.debug("apply_configured_global_mask is disabled; use apply_configured_stream_mask")
    return False


def apply_configured_stream_mask(stream_or_ptr: object) -> bool:
    global _stream_mask_touch_count
    mask = get_configured_stream_mask()
    if mask is None or mask == 0:
        return False

    if isinstance(stream_or_ptr, int):
        stream_ptr = stream_or_ptr
    else:
        stream_ptr = int(getattr(stream_or_ptr, "cuda_stream"))

    if stream_ptr == 0:
        return False

    with _mask_lock:
        _stream_mask_touch_count += 1
        _append_stream_mask_touch_log(ts_ns=time.monotonic_ns())
        prev_mask = _masked_stream_masks.get(stream_ptr)
        if prev_mask == mask:
            return True
        set_stream_mask(stream_ptr, mask)
        _masked_stream_masks[stream_ptr] = mask
        return True
