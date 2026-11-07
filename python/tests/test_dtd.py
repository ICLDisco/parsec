# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

import ctypes
import gc
import threading

import numpy as np
import pytest

import py_parsec
from py_parsec import dtd

from conftest import assert_matches, scatter


def test_gemm(ctx, tp):
    """C += A @ B with Python bodies; tiles move between ranks."""
    M, N, K, b = 64, 48, 80, 16
    rng = np.random.default_rng(7)
    Ag, Bg, Cg = rng.standard_normal((M, K)), rng.standard_normal((K, N)), rng.standard_normal((M, N))
    A = py_parsec.BlockCyclic(ctx, M, K, b, b, name="A")
    B = py_parsec.BlockCyclic(ctx, K, N, b, b, name="B")
    C = py_parsec.BlockCyclic(ctx, M, N, b, b, name="C")
    for X, G in ((A, Ag), (B, Bg), (C, Cg)):
        scatter(X, G)

    gemm = tp.task_class("gemm", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.INPUT, B),
                         dtd.Data(dtd.INOUT | dtd.AFFINITY, C))

    def body(a, b, c):
        c += a @ b

    gemm.add_chore(py_parsec.DEV_CPU, body)
    for m in range(C.mt):
        for n in range(C.nt):
            for k in range(A.nt):
                tp.insert_task(gemm, dtd.tile_of(A, m, k), dtd.tile_of(B, k, n), dtd.tile_of(C, m, n))
    for X in (A, B, C):
        tp.data_flush_all(X)
    tp.wait()
    assert_matches(C, Cg + Ag @ Bg, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128, np.int32])
def test_transpose_tiles(ctx, tp, dtype):
    """B(m, n) = A(n, m).T moves tiles of every supported type across ranks."""
    n, b = 24, 8
    full = (np.arange(n * n).reshape(n, n) % 251).astype(dtype)
    if np.iscomplexobj(full):
        full = full + 1j * full.T
    A = py_parsec.BlockCyclic(ctx, n, n, b, b, dtype=dtype, name="A")
    B = py_parsec.BlockCyclic(ctx, n, n, b, b, dtype=dtype, name="B")
    scatter(A, full)

    tc = tp.task_class("transpose", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))

    def body(a, out):
        out[:] = a.T

    tc.add_chore(py_parsec.DEV_CPU, body)
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(tc, dtd.tile_of(A, j, i), dtd.tile_of(B, i, j))
    tp.data_flush_all(A)
    tp.data_flush_all(B)
    tp.wait()
    assert_matches(B, full.T)


def test_wait_flushes_used_collections(ctx, tp):
    """Without data_flush_all, wait() flushes the collections the tasks used
    (a distributed DTD wait would not terminate otherwise)."""
    n, b = 24, 8
    full = np.arange(n * n, dtype=np.float64).reshape(n, n)
    A = py_parsec.BlockCyclic(ctx, n, n, b, b, name="A")
    B = py_parsec.BlockCyclic(ctx, n, n, b, b, name="B")
    scatter(A, full)
    tc = tp.task_class("transpose", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))
    tc.add_chore(py_parsec.DEV_CPU, lambda a, out: np.copyto(out, a.T))
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(tc, dtd.tile_of(A, j, i), dtd.tile_of(B, i, j))
    tp.wait()
    assert_matches(B, full.T)
    copy = tp.task_class("copy", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))
    copy.add_chore(py_parsec.DEV_CPU, lambda a, out: np.copyto(out, a))
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(copy, dtd.tile_of(A, i, j), dtd.tile_of(B, i, j))
    ctx.wait()
    assert_matches(B, full)


def test_values(ctx, tp):
    formats = "bBhHiIlLqQfd"
    values = [-5, 250, -300, 60000, -70000, 4000000000, -2**40, 2**41, -2**60, 2**63 + 5, 1.5, -2.25]
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)
    tc = tp.task_class("values", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), *(dtd.Value(f) for f in formats))
    seen = []
    tc.add_chore(py_parsec.DEV_CPU, lambda tile, *args: seen.append(args))
    tp.insert_task(tc, dtd.tile_of(A, 0, 0), *values)
    with pytest.raises(OverflowError):
        tp.insert_task(tc, dtd.tile_of(A, 0, 0), 1000, *values[1:])
    tp.data_flush_all(A)
    tp.wait()
    assert seen == ([tuple(values)] if A.is_local(0, 0) else [])


def test_body_exception_raised_by_wait(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    tc = tp.task_class("boom", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))

    def body(tile):
        raise ZeroDivisionError("raised in a task body")

    tc.add_chore(py_parsec.DEV_CPU, body)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, dtd.tile_of(A, m, n))
    tp.data_flush_all(A)
    if A.nb_local_tiles:
        with pytest.raises(ZeroDivisionError, match="raised in a task body"):
            tp.wait()
    else:
        tp.wait()
    tp.wait()  # reported once


def test_c_function_chore(ctx, tp):
    """A chore given as a C function pointer (here a ctypes callback)."""
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8)
    calls = []
    lock = threading.Lock()

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def chore(es, task):
        with lock:
            calls.append(task)
        return py_parsec.HOOK_RETURN_DONE

    tc = tp.task_class("c_chore", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, chore)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, dtd.tile_of(A, m, n))
    tp.data_flush_all(A)
    tp.wait()
    assert len(calls) == A.nb_local_tiles


def test_several_waits(ctx, tp):
    """A DTD taskpool can be fed again after a wait."""
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda t: t.__iadd__(1))
    for _ in range(3):
        for m in range(A.mt):
            for n in range(A.nt):
                tp.insert_task(inc, dtd.tile_of(A, m, n))
        tp.data_flush_all(A)
        tp.wait()
    assert_matches(A, np.full((16, 16), 3.0))


def test_insert_errors(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    small = py_parsec.BlockCyclic(ctx, 16, 16, 4, 4)
    tc = tp.task_class("t", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("i"))
    with pytest.raises(ValueError, match="no chore"):
        tp.insert_task(tc, dtd.tile_of(A, 0, 0), 1)
    tc.add_chore(py_parsec.DEV_CPU, lambda t, i: None)
    with pytest.raises(ValueError, match="already has"):
        tc.add_chore(py_parsec.DEV_CPU, lambda t, i: None)
    with pytest.raises(ValueError, match="exactly one device type"):
        tc.add_chore(py_parsec.DEV_ALL, lambda t, i: None)
    if ctx.nb_devices(py_parsec.DEV_CUDA):
        with pytest.raises(TypeError, match="DEV_CPU"):
            tc.add_chore(py_parsec.DEV_CUDA, lambda t, i: None)
    else:
        with pytest.raises(ValueError, match="no device"):
            tc.add_chore(py_parsec.DEV_CUDA, 0x1234)
    with pytest.raises(TypeError, match="takes 2 arguments"):
        tp.insert_task(tc, dtd.tile_of(A, 0, 0))
    with pytest.raises(TypeError, match="must be a Tile"):
        tp.insert_task(tc, 0, 1)
    with pytest.raises(TypeError):
        tp.insert_task(tc, dtd.tile_of(A, 0, 0), "one")
    with pytest.raises(ValueError, match="expects 8x8"):
        tp.insert_task(tc, dtd.tile_of(small, 0, 0), 1)
    with pytest.raises(ValueError, match="no chore for device"):
        tp.insert_task(tc, dtd.tile_of(A, 0, 0), 1, device=py_parsec.DEV_CUDA)
    with pytest.raises(IndexError):
        dtd.tile_of(A, 5, 0)

    other = dtd.Taskpool()
    with pytest.raises(RuntimeError, match="context first"):
        other.task_class("x", dtd.Data(dtd.INPUT, A))
    ctx.add_taskpool(other)
    with pytest.raises(ValueError, match="another taskpool"):
        other.insert_task(tc, dtd.tile_of(A, 0, 0), 1)
    other.free()


def test_param_errors(ctx):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    with pytest.raises(ValueError, match="INPUT, OUTPUT or INOUT"):
        dtd.Data(dtd.AFFINITY, A)
    with pytest.raises(TypeError):
        dtd.Data(dtd.INPUT)
    with pytest.raises(ValueError, match="format"):
        dtd.Value("x")
    d = dtd.Data(dtd.INPUT, shape=(4, 2), dtype=np.float32)
    assert (d.mb, d.nb, d.dtype) == (4, 2, np.float32)
    with pytest.raises(ValueError, match="flags"):
        dtd.tile_of(A, 0, 0, flags=dtd.INPUT)


def test_garbage_collected_without_free(ctx):
    """Taskpools dropped without free() are released by the garbage collector."""
    ctx.start()
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    plain = dtd.Taskpool()
    ctx.add_taskpool(plain)
    plain.wait()
    del plain                       # no task class: freed by the refcount

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def chore(es, task):
        return py_parsec.HOOK_RETURN_DONE

    with_classes = dtd.Taskpool()
    ctx.add_taskpool(with_classes)
    tc = with_classes.task_class("c", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, chore)
    with_classes.insert_task(tc, dtd.tile_of(A, 0, 0))
    with_classes.wait()
    del with_classes, tc            # taskpool <-> task class cycle: the GC frees it
    never_added = dtd.Taskpool()
    del never_added
    gc.collect()


def test_write_without_affinity(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    tc = tp.task_class("w", dtd.Data(dtd.INOUT, A))
    tc.add_chore(py_parsec.DEV_CPU, lambda t: None)
    if ctx.nb_nodes > 1:
        with pytest.raises(ValueError, match="AFFINITY"):
            tp.insert_task(tc, dtd.tile_of(A, 0, 0))
    else:
        tp.insert_task(tc, dtd.tile_of(A, 0, 0))
        tp.wait()


def test_body_return_codes(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    again = tp.task_class("again", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    calls = []

    def body(tile):
        calls.append(1)
        return py_parsec.HOOK_RETURN_AGAIN if len(calls) == 1 else None

    again.add_chore(py_parsec.DEV_CPU, body)
    tp.insert_task(again, dtd.tile_of(A, 0, 0))
    tp.wait()
    assert len(calls) == (2 if A.is_local(0, 0) else 0)

    bad = tp.task_class("bad", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    bad.add_chore(py_parsec.DEV_CPU, lambda t: 7)
    tp.insert_task(bad, dtd.tile_of(A, 1, 1))
    if A.is_local(1, 1):
        with pytest.raises(ValueError, match="returned 7"):
            tp.wait()
    else:
        tp.wait()


def test_too_many_flows(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    assert 1 <= dtd.MAX_FLOWS <= dtd.MAX_PARAMS
    tp.task_class("max", *[dtd.Data(dtd.INPUT, A)] * dtd.MAX_FLOWS)
    with pytest.raises(ValueError, match="Data parameters"):
        tp.task_class("too_many", *[dtd.Data(dtd.INPUT, A)] * (dtd.MAX_FLOWS + 1))


def test_dropped_with_pending_work_completes(ctx):
    """A taskpool dropped before wait() completes its tasks before it is freed."""
    ctx.start()
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8)
    calls = []
    lock = threading.Lock()

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def chore(es, task):
        with lock:
            calls.append(1)
        return py_parsec.HOOK_RETURN_DONE

    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    tc = tp.task_class("count", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, chore)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, dtd.tile_of(A, m, n))
    del tp, tc
    gc.collect()
    assert len(calls) == A.nb_local_tiles
