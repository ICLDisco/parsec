"""
Py_PaRSEC: Python interface for PaRSEC
"""

__version__ = "0.1.0"

# --- DTD module (what you actually have today) ---
from . import dtd as _dtd

def _lazy_get(mod, primary, fallback=None):
    v = getattr(mod, primary, None)
    if v is not None:
        return v
    if fallback is None:
        raise AttributeError(f"{mod.__name__} has no attribute {primary}")
    return getattr(mod, fallback)

# DTD 的就只取 DTD 的（不存在就报错，不要去碰 ParsecContext）
ParsecDTDContext  = _lazy_get(_dtd, "ParsecDTDContext")
ParsecDTDTaskpool = _lazy_get(_dtd, "ParsecDTDTaskpool")
try:
    ParsecDTDMatrix   = _lazy_get(_dtd, "ParsecDTDMatrix")
except AttributeError:
    ParsecDTDMatrix = None

PARSEC_INPUT    = _dtd.PARSEC_INPUT
PARSEC_INOUT    = _dtd.PARSEC_INOUT
PARSEC_VALUE    = _dtd.PARSEC_VALUE
PARSEC_AFFINITY = _dtd.PARSEC_AFFINITY
PARSEC_PUSHOUT  = _dtd.PARSEC_PUSHOUT

PARSEC_DTD_EMPTY_FLAG = _dtd.PARSEC_DTD_EMPTY_FLAG
PARSEC_DTD_ARG_END    = _dtd.PARSEC_DTD_ARG_END

PARSEC_DEV_CPU  = _dtd.PARSEC_DEV_CPU
PARSEC_DEV_CUDA = _dtd.PARSEC_DEV_CUDA

parsec_redistribute_dtd = _lazy_get(_dtd, "parsec_redistribute_dtd")
parsec_redistribute = _lazy_get(_dtd, "parsec_redistribute")

# --- Matrix module (what you actually have today) ---
from .matrix import ParsecMatrixBlockCyclic

# 给你一个“名字兼容”，先别再导出 None
ParsecDTDMatrix = ParsecMatrixBlockCyclic

__all__ = [
    "ParsecDTDContext",
    "ParsecDTDTaskpool",
    "ParsecDTDTaskClass",

    "ParsecMatrixBlockCyclic",
    "ParsecDTDMatrix",

    "PARSEC_INPUT",
    "PARSEC_INOUT",
    "PARSEC_VALUE",
    "PARSEC_AFFINITY",
    "PARSEC_PUSHOUT",
    "PARSEC_DTD_EMPTY_FLAG",
    "PARSEC_DTD_ARG_END",
    "PARSEC_DEV_CPU",
    "PARSEC_DEV_CUDA",
    "parsec_redistribute_dtd",
    "parsec_redistribute",
]
