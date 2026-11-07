# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

import pytest

import py_parsec
from py_parsec import dtd


def test_version():
    major, minor, release = py_parsec.version()
    assert major >= 4


def test_context(ctx):
    assert 0 <= ctx.rank < ctx.nb_nodes
    assert ctx.nb_cores >= 1
    assert ctx.address != 0
    assert ctx.nb_devices(py_parsec.DEV_CPU) >= 1
    assert "Context" in repr(ctx)


def test_start_and_wait(ctx):
    ctx.start()
    assert ctx.active
    assert not ctx.start()  # already running
    ctx.wait()
    assert not ctx.active
    with pytest.raises(RuntimeError, match="not been started"):
        ctx.wait()


def test_taskpool_lifecycle(ctx):
    tp = dtd.Taskpool()
    with pytest.raises(RuntimeError, match="not been added"):
        tp.wait()
    ctx.add_taskpool(tp)
    assert tp.context is ctx
    with pytest.raises(RuntimeError, match="already been added"):
        ctx.add_taskpool(tp)
    with pytest.raises(RuntimeError, match="not been started"):
        tp.wait()
    ctx.start()
    tp.wait()
    tp.free()
    with pytest.raises(RuntimeError, match="released"):
        tp.wait()
    tp.free()  # idempotent


def test_context_wait_completes_dtd_taskpools(ctx):
    ctx.start()
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    ctx.wait()  # waits the DTD taskpool first
    tp.free()


def test_taskpool_from_address_rejects_null():
    with pytest.raises(ValueError):
        py_parsec.Taskpool.from_address(0)
