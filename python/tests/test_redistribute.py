# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""PTG and DTD taskpools on the same context and the same data."""

import gc

import numpy as np
import pytest

import py_parsec
from py_parsec import dtd
from py_parsec.matrix import redistribute_dtd, redistribute_new

from conftest import assert_matches, scatter

M, N = 50, 40


def global_matrix():
    return np.arange(M * N, dtype=np.float64).reshape(M, N)


def test_redistribute_ptg(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    tp = redistribute_new(src, dst)
    ctx.add_taskpool(tp)
    ctx.start()
    ctx.wait()
    tp.free()
    assert_matches(dst, global_matrix())


def test_redistribute_ptg_submatrix(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, 64, 64, 16, 16, name="dst")
    scatter(src, global_matrix())
    ctx.start()
    tp = redistribute_new(src, dst, 20, 15, disi_source=5, disj_source=3, disi_target=10, disj_target=7)
    ctx.add_taskpool(tp)
    ctx.wait()
    tp.free()
    expected = np.zeros((64, 64))
    expected[10:30, 7:22] = global_matrix()[5:25, 3:18]
    assert_matches(dst, expected)


def test_redistribute_ptg_arguments(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    with pytest.raises(TypeError, match="dtypes differ"):
        redistribute_new(src, py_parsec.BlockCyclic(ctx, M, N, 8, 8, dtype=np.float32))
    with pytest.raises(ValueError, match="rejected"):
        redistribute_new(src, py_parsec.BlockCyclic(ctx, 10, 10, 8, 8))


def test_redistribute_dtd(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    redistribute_dtd(ctx, src, dst)
    assert not ctx.active
    assert_matches(dst, global_matrix())


def test_dtd_then_ptg_then_dtd(ctx):
    """DTD writes src, a PTG taskpool copies it to dst, DTD scales dst."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 10, 10, P=1, name="dst")
    ctx.start()

    fill = dtd.Taskpool()
    ctx.add_taskpool(fill)
    init = fill.task_class("init", dtd.Data(dtd.OUTPUT | dtd.AFFINITY, src), dtd.Value("i"), dtd.Value("i"))

    def init_body(tile, m, n):
        rows = np.arange(m * 8, m * 8 + 8)[:, None]
        cols = np.arange(n * 8, n * 8 + 8)[None, :]
        tile[:] = rows * N + cols

    init.add_chore(py_parsec.DEV_CPU, init_body)
    for m in range(src.mt):
        for n in range(src.nt):
            fill.insert_task(init, dtd.tile_of(src, m, n), m, n)
    fill.data_flush_all(src)
    ctx.wait()

    copy = redistribute_new(src, dst)
    ctx.add_taskpool(copy)
    ctx.start()
    ctx.wait()

    ctx.start()
    scale = dtd.Taskpool()
    ctx.add_taskpool(scale)
    twice = scale.task_class("twice", dtd.Data(dtd.INOUT | dtd.AFFINITY, dst))
    twice.add_chore(py_parsec.DEV_CPU, lambda tile: tile.__imul__(2))
    for m in range(dst.mt):
        for n in range(dst.nt):
            scale.insert_task(twice, dtd.tile_of(dst, m, n))
    scale.data_flush_all(dst)
    ctx.wait()
    for t in (fill, copy, scale):
        t.free()
    assert_matches(dst, 2 * global_matrix())


def test_never_enqueued_taskpool(ctx):
    """PaRSEC cannot release a PTG taskpool that never ran: it is leaked with a warning."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    dst = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    tp = redistribute_new(src, dst)
    with pytest.warns(ResourceWarning, match="never enqueued"):
        tp.free()
    dropped = redistribute_new(src, dst)
    del dropped
    gc.collect()
