# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Dynamic Task Discovery (DTD).

A DTD taskpool is an ordinary taskpool: it is added to a
:class:`py_parsec.Context` with ``add_taskpool`` like any other. Task classes
mirror ``parsec_dtd_create_task_class``: each parameter is either a data tile
(:class:`Data`) or a value (:class:`Value`). Chores are Python callables
(CPU only) or C function pointers (any device). A Python body receives each
tile as an ``(mb, nb)`` column-major NumPy array that aliases the PaRSEC data
copy, and each value as a Python number, in declaration order::

    import py_parsec
    from py_parsec import dtd

    with py_parsec.Context() as ctx:
        A = py_parsec.BlockCyclic(ctx, 1000, 1000, 100, 100)
        tp = dtd.Taskpool()
        ctx.add_taskpool(tp)
        ctx.start()

        def body(tile, alpha):
            tile *= alpha

        scale = tp.task_class("scale", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("d"))
        scale.add_chore(py_parsec.DEV_CPU, body)
        for m in range(A.mt):
            for n in range(A.nt):
                tp.insert_task(scale, dtd.tile_of(A, m, n), 2.0)
        tp.wait()   # flushes A, then completes the tasks

Python bodies hold the GIL while they run, so CPU parallelism comes from
the code they call releasing it (NumPy, SciPy, ...). Keep the BLAS under
NumPy to one thread (``OPENBLAS_NUM_THREADS=1``) and give each body about a
millisecond of work or more; short bodies spend their time waiting for the
GIL when many workers share a process. The arrays handed to a body are only
valid during the call. A body may return ``py_parsec.HOOK_RETURN_AGAIN``
to be rescheduled later, or ``HOOK_RETURN_DONE``; any other integer is an
error, and any other result (including None) means the task is done. An
exception raised by a body is printed, the task is completed, and the
exception is raised again by the next ``wait`` on the taskpool.
"""

import ctypes
import traceback
import weakref

cimport cython
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, int64_t, uint64_t, uintptr_t

import numpy as np

from py_parsec cimport cparsec as c
from py_parsec.matrix cimport TiledMatrix, host_array
from py_parsec.runtime cimport Context
from py_parsec.runtime cimport Taskpool as BaseTaskpool
from py_parsec.runtime import address_of, runtime_initialized
from py_parsec.matrix import matrix_type

INPUT = c.PARSEC_INPUT
OUTPUT = c.PARSEC_OUTPUT
INOUT = c.PARSEC_INOUT
AFFINITY = c.PARSEC_AFFINITY
DONT_TRACK = c.PARSEC_DONT_TRACK
PUSHOUT = c.PARSEC_PUSHOUT
PULLIN = c.PARSEC_PULLIN

#: Largest number of parameters of a task class.
MAX_PARAMS = c.PARSEC_DTD_MAX_PARAMS
#: Largest number of Data parameters (flows) of a task class.
MAX_FLOWS = c.pyparsec_dtd_max_flows()

cdef int _ACCESS_MODES = c.PARSEC_INPUT | c.PARSEC_OUTPUT | c.PARSEC_INOUT
cdef int _DATA_FLAGS = c.PARSEC_AFFINITY | c.PARSEC_DONT_TRACK | c.PARSEC_PUSHOUT | c.PARSEC_PULLIN


cdef class Param:
    """A parameter of a task class (see :class:`Data` and :class:`Value`)."""

    cdef int size(self):
        return 0

    cdef object load(self, uint64_t *slot):
        raise NotImplementedError

    cdef int store(self, object value, uint64_t *slot) except -1:
        raise NotImplementedError


cdef class Data(Param):
    """A tile parameter.

    Parameters
    ----------
    op : int
        ``INPUT``, ``OUTPUT`` or ``INOUT``, possibly or-ed with ``AFFINITY``
        (run the task where this tile lives), ``DONT_TRACK``, ``PUSHOUT`` or
        ``PULLIN``.
    like : TiledMatrix, optional
        A matrix whose tiles go through this parameter; it provides the tile
        shape and dtype. Otherwise give ``shape=(mb, nb)`` and ``dtype``.
    """

    cdef readonly int op
    cdef readonly int mb
    cdef readonly int nb
    cdef readonly object dtype
    cdef int mtype
    cdef int typenum

    def __init__(self, int op, TiledMatrix like=None, *, shape=None, dtype=None):
        if (op & c.PARSEC_GET_OP_TYPE) not in (c.PARSEC_INPUT, c.PARSEC_OUTPUT, c.PARSEC_INOUT):
            raise ValueError("a Data parameter needs INPUT, OUTPUT or INOUT")
        if op & ~(_ACCESS_MODES | _DATA_FLAGS):
            raise ValueError(f"unsupported flags {op:#x} for a Data parameter")
        if like is not None:
            if shape is not None or dtype is not None:
                raise TypeError("give either like= or shape= and dtype=")
            self.mb, self.nb, self.dtype = like.mb, like.nb, like.dtype
        else:
            if shape is None or dtype is None:
                raise TypeError("a Data parameter needs like= or shape= and dtype=")
            self.mb, self.nb = shape
            self.dtype = np.dtype(dtype)
        self.op = op
        self.mtype = matrix_type(self.dtype)
        self.typenum = self.dtype.num

    cdef object load(self, uint64_t *slot):
        return host_array((<void **>slot)[0], self.mb, self.nb, self.typenum, None)

    def __repr__(self):
        return f"Data({self.op:#x}, shape=({self.mb}, {self.nb}), dtype={self.dtype})"


cdef class Value(Param):
    """A parameter passed by value, given as a :mod:`struct` format character.

    One of ``b B h H i I l L q Q f d`` (native C char ... double); the
    default ``"i"`` is a C ``int``.
    """

    cdef readonly str format
    cdef int nbytes
    cdef char code

    def __init__(self, str format="i"):
        sizes = {"b": sizeof(int8_t), "B": sizeof(uint8_t), "h": sizeof(int16_t),
                 "H": sizeof(uint16_t), "i": sizeof(int), "I": sizeof(unsigned int),
                 "l": sizeof(long), "L": sizeof(unsigned long), "q": sizeof(int64_t),
                 "Q": sizeof(uint64_t), "f": sizeof(float), "d": sizeof(double)}
        if format not in sizes:
            raise ValueError(f"unsupported Value format {format!r}")
        self.format = format
        self.nbytes = sizes[format]
        self.code = ord(format)

    cdef int size(self):
        return self.nbytes

    cdef object load(self, uint64_t *slot):
        cdef void *p = <void *>slot
        if self.code == b'b': return (<int8_t *>p)[0]
        if self.code == b'B': return (<uint8_t *>p)[0]
        if self.code == b'h': return (<int16_t *>p)[0]
        if self.code == b'H': return (<uint16_t *>p)[0]
        if self.code == b'i': return (<int *>p)[0]
        if self.code == b'I': return (<unsigned int *>p)[0]
        if self.code == b'l': return (<long *>p)[0]
        if self.code == b'L': return (<unsigned long *>p)[0]
        if self.code == b'q': return (<int64_t *>p)[0]
        if self.code == b'Q': return (<uint64_t *>p)[0]
        if self.code == b'f': return (<float *>p)[0]
        return (<double *>p)[0]

    cdef int store(self, object value, uint64_t *slot) except -1:
        cdef void *p = <void *>slot
        slot[0] = 0
        if self.code == b'b': (<int8_t *>p)[0] = value
        elif self.code == b'B': (<uint8_t *>p)[0] = value
        elif self.code == b'h': (<int16_t *>p)[0] = value
        elif self.code == b'H': (<uint16_t *>p)[0] = value
        elif self.code == b'i': (<int *>p)[0] = value
        elif self.code == b'I': (<unsigned int *>p)[0] = value
        elif self.code == b'l': (<long *>p)[0] = value
        elif self.code == b'L': (<unsigned long *>p)[0] = value
        elif self.code == b'q': (<int64_t *>p)[0] = value
        elif self.code == b'Q': (<uint64_t *>p)[0] = value
        elif self.code == b'f': (<float *>p)[0] = value
        else: (<double *>p)[0] = value
        return 0

    def __repr__(self):
        return f"Value({self.format!r})"


cdef class Tile:
    """Tile (m, n) of a data collection, as passed to :meth:`Taskpool.insert_task`.

    ``flags`` are added to the access mode declared by the task class for
    this insertion only (for instance ``PUSHOUT`` on the last update).
    """

    cdef readonly TiledMatrix matrix
    cdef readonly int m
    cdef readonly int n
    cdef readonly int flags
    cdef c.parsec_data_key_t key

    def __repr__(self):
        return f"tile_of({self.matrix.name}, {self.m}, {self.n}, flags={self.flags:#x})"


def tile_of(TiledMatrix matrix not None, int m, int n, int flags=0):
    """Reference to tile (m, n) of ``matrix`` for a DTD insertion
    (the ``PARSEC_DTD_TILE_OF`` of the C interface)."""
    cdef Tile t
    if flags & ~_DATA_FLAGS:
        raise ValueError(f"unsupported insertion flags {flags:#x}")
    matrix.rank_of(m, n)  # bounds check
    t = Tile.__new__(Tile)
    t.matrix = matrix
    t.m = m
    t.n = n
    t.flags = flags
    t.key = c.pyparsec_data_key(matrix.dc(), m, n)
    return t


# Task classes with a Python chore, indexed by their parsec_task_class_t
# address, for the chore trampoline below.
cdef dict _python_task_classes = {}


cdef c.parsec_hook_return_t _python_chore(c.parsec_execution_stream_t *es,
                                          c.parsec_task_t *task) noexcept with gil:
    cdef TaskClass tc = _python_task_classes.get(<uintptr_t>c.pyparsec_task_class_of(task))
    if tc is None:
        return c.PARSEC_HOOK_RETURN_ERROR
    return tc.execute(task)


cdef class TaskClass:
    """A DTD task class; create it with :meth:`Taskpool.task_class`."""

    cdef c.parsec_task_class_t *_tc
    cdef object _taskpool_ref   # weak reference: a task class never keeps its taskpool alive
    cdef bytes _name
    cdef readonly tuple params
    cdef int _devices
    cdef object _body
    cdef list _keepalive
    cdef object _error

    @property
    def name(self):
        return self._name.decode()

    @property
    def devices(self):
        """Mask of the device types that have a chore."""
        return self._devices

    cdef Taskpool taskpool(self):
        """The taskpool of the task class, or None once it is released."""
        if self._taskpool_ref is None:
            return None
        return self._taskpool_ref()

    cdef c.parsec_task_class_t *create(self, Taskpool tp) except NULL:
        """Create the C task class (deferred until its first chore)."""
        cdef int sizes[c.PARSEC_DTD_MAX_PARAMS]
        cdef int ops[c.PARSEC_DTD_MAX_PARAMS]
        cdef int i
        cdef Param p
        cdef Context ctx = tp.context
        for i, p in enumerate(self.params):
            if isinstance(p, Data):
                sizes[i] = c.PASSED_BY_REF
                ops[i] = (<Data>p).op | ctx.dtd_tile_arena((<Data>p).mtype, (<Data>p).mb,
                                                           (<Data>p).nb, (<Data>p).mb)
            else:
                sizes[i] = p.size()
                ops[i] = c.PARSEC_VALUE
        self._tc = c.pyparsec_dtd_create_task_class(self._name, len(self.params), sizes, ops)
        if self._tc == NULL:
            raise RuntimeError(f"cannot create task class {self.name}")
        return self._tc

    def add_chore(self, int device_type, body):
        """Add the body run on ``device_type`` (``parsec_dtd_task_class_add_chore``).

        ``body`` is either a Python callable, for ``DEV_CPU`` only, or the
        address of a C function (int, unnamed PyCapsule or ctypes function
        pointer) with the DTD chore signature of that device type.
        """
        cdef void *fn
        cdef int rc
        cdef uintptr_t key
        cdef Taskpool tp = self.taskpool()
        if tp is None or tp._tp == NULL:
            raise RuntimeError("the taskpool of this task class has been released")
        if (device_type <= 0 or device_type & ~c.PARSEC_DEV_ALL
                or device_type & (device_type - 1)):
            raise ValueError(f"a chore runs on exactly one device type, not {device_type:#x}")
        if device_type & self._devices:
            raise ValueError(f"task class {self.name} already has a chore for device {device_type:#x}")
        if tp.context.nb_devices(device_type) == 0:
            raise ValueError(f"the context has no device of type {device_type:#x}")

        if isinstance(body, ctypes._CFuncPtr):
            fn = <void *><uintptr_t>ctypes.cast(body, ctypes.c_void_p).value
            self._keepalive.append(body)
        elif callable(body):
            if device_type != c.PARSEC_DEV_CPU:
                raise TypeError("a Python body runs on DEV_CPU; accelerator chores take a C function address")
            if self._body is not None:
                raise ValueError(f"task class {self.name} already has a Python body")
            fn = <void *>_python_chore
        else:
            fn = <void *><uintptr_t>address_of(body)
            if fn == NULL:
                raise ValueError("NULL chore")

        if self._tc == NULL:
            self.create(tp)
        key = <uintptr_t>self._tc
        if fn == <void *>_python_chore:
            self._body = body
            _python_task_classes[key] = self
        rc = c.parsec_dtd_task_class_add_chore(tp._tp, self._tc, device_type, fn)
        if rc != c.PARSEC_SUCCESS:
            if fn == <void *>_python_chore:
                self._body = None
                del _python_task_classes[key]
            if self._devices == 0:
                # Never registered in the taskpool: releasing it is not
                # supported by DTD, leave it be.
                self._tc = NULL
            raise RuntimeError(f"cannot add a chore for device {device_type:#x} to {self.name} ({rc})")
        self._devices |= device_type

    cdef c.parsec_hook_return_t execute(self, c.parsec_task_t *task) noexcept:
        """Run the Python body of a task; called with the GIL held."""
        cdef uint64_t slots[c.PARSEC_DTD_MAX_PARAMS]
        cdef void *outs[c.PARSEC_DTD_MAX_PARAMS]
        cdef int i, nparams = len(self.params)
        for i in range(nparams):
            slots[i] = 0
            outs[i] = &slots[i]
        c.pyparsec_dtd_unpack(task, nparams, outs)
        try:
            args = [(<Param>self.params[i]).load(&slots[i]) for i in range(nparams)]
            rc = self._body(*args)
            if rc is None or not isinstance(rc, int) or isinstance(rc, bool):
                return c.PARSEC_HOOK_RETURN_DONE
            if rc == c.PARSEC_HOOK_RETURN_DONE or rc == c.PARSEC_HOOK_RETURN_AGAIN:
                return <c.parsec_hook_return_t><int>rc
            raise ValueError(f"the body of {self.name} returned {rc}: only HOOK_RETURN_DONE "
                             "and HOOK_RETURN_AGAIN are accepted")
        except BaseException as e:
            traceback.print_exc()
            if self._error is None:
                self._error = e
            return c.PARSEC_HOOK_RETURN_DONE

    cdef release(self, c.parsec_taskpool_t *tp):
        if self._tc != NULL:
            _python_task_classes.pop(<uintptr_t>self._tc, None)
            c.parsec_dtd_task_class_release(tp, self._tc)
            self._tc = NULL
        self._taskpool_ref = None
        self._body = None
        self._keepalive = []

    def __repr__(self):
        return f"<py_parsec.dtd.TaskClass {self.name!r} {list(self.params)}>"


@cython.no_gc_clear
cdef class Taskpool(BaseTaskpool):
    """A DTD taskpool (``parsec_dtd_taskpool_new``).

    Insert tasks once the taskpool has been added to a context; waiting on
    it (or on the context) completes the tasks inserted so far, and more can
    be inserted afterwards. DTD needs every collection used by the tasks to
    be flushed before a wait: :meth:`wait` and :meth:`Context.wait` first
    call :meth:`data_flush_all` on the collections used since their last
    flush, so that their data is back on its owners when the wait returns.
    """

    cdef list _classes
    cdef dict _collections   # TiledMatrix -> used since its last flush; also
                             # keeps them alive while tasks may refer to them

    def __cinit__(self):
        self._classes = []
        self._collections = {}
        if not runtime_initialized():
            raise RuntimeError("create a py_parsec.Context before a DTD taskpool")
        self._tp = c.parsec_dtd_taskpool_new()
        if self._tp == NULL:
            raise RuntimeError("parsec_dtd_taskpool_new failed")

    def __dealloc__(self):
        # Runs before the attributes of this class are cleared and before the
        # base __dealloc__. Pending work is completed first, as in free(),
        # because the matrices the tasks use may be freed right after this;
        # then the task classes go with the taskpool. Python bodies cannot
        # reach this point: _python_task_classes keeps their taskpool alive.
        cdef TaskClass tc
        if self._tp == NULL:
            return
        if self._context is not None:
            if self._context._ctx == NULL:
                self._tp = NULL         # the context is finalized
                return
            if self._needs_wait:
                try:
                    if not self._context.is_active():
                        self._context.start()
                    BaseTaskpool.wait(self)
                except BaseException:
                    traceback.print_exc()
                if self._needs_wait:
                    self._tp = NULL     # could not be completed: leaked
                    return
        elif not runtime_initialized():
            self._tp = NULL             # never enqueued and no runtime left
            return
        for tc in self._classes:
            tc.release(self._tp)
        self._destruct(self._tp)
        self._tp = NULL

    cdef Context attached_context(self):
        self.ptr()
        if self._context is None:
            raise RuntimeError("add the taskpool to a context first")
        return self._context

    def task_class(self, name, *params):
        """Create a task class whose parameters are :class:`Data` and :class:`Value`.

        The taskpool must already have been added to a context.
        """
        cdef TaskClass tc
        self.attached_context()
        if len(params) > c.PARSEC_DTD_MAX_PARAMS:
            raise ValueError(f"a task class has at most {c.PARSEC_DTD_MAX_PARAMS} parameters")
        for p in params:
            if not isinstance(p, Param):
                raise TypeError(f"task class parameters are Data or Value, not {type(p).__name__}")
        if sum(isinstance(p, Data) for p in params) > MAX_FLOWS:
            raise ValueError(f"a task class has at most {MAX_FLOWS} Data parameters")
        tc = TaskClass.__new__(TaskClass)
        tc._taskpool_ref = weakref.ref(self)
        tc._name = str(name).encode()
        tc.params = params
        tc._keepalive = []
        self._classes.append(tc)
        return tc

    def insert_task(self, TaskClass tc not None, *args, int priority=0, int device=c.PARSEC_DEV_ALL):
        """Insert a task of ``tc`` (``parsec_dtd_insert_task_with_task_class``).

        Data parameters take a :class:`Tile` (see :func:`tile_of`), value
        parameters a Python number. ``device`` restricts the chores the task
        may run with.
        """
        cdef int flags[c.PARSEC_DTD_MAX_PARAMS]
        cdef void *ptrs[c.PARSEC_DTD_MAX_PARAMS]
        cdef uint64_t values[c.PARSEC_DTD_MAX_PARAMS]
        cdef int i, nargs = len(args), write_flows = 0
        cdef bint affinity = False
        cdef Param p
        cdef Data d
        cdef Tile t
        cdef Context ctx = self.attached_context()
        cdef c.parsec_taskpool_t *tp = self._tp
        cdef c.parsec_task_class_t *ctc

        if tc.taskpool() is not self:
            raise ValueError(f"task class {tc.name} belongs to another taskpool")
        if not (device & tc._devices):
            raise ValueError(f"task class {tc.name} has no chore for device mask {device:#x}")
        if nargs != len(tc.params):
            raise TypeError(f"task class {tc.name} takes {len(tc.params)} arguments, {nargs} given")

        # Check and convert every argument before any DTD tile is created: a
        # tile handle left behind by a failed insertion could not be flushed.
        for i in range(nargs):
            p = tc.params[i]
            if isinstance(p, Data):
                d = <Data>p
                if not isinstance(args[i], Tile):
                    raise TypeError(f"argument {i} of {tc.name} must be a Tile (see tile_of)")
                t = <Tile>args[i]
                if t.matrix.mb != d.mb or t.matrix.nb != d.nb or t.matrix.dtype != d.dtype:
                    raise ValueError(f"argument {i} of {tc.name}: {t.matrix.mb}x{t.matrix.nb} "
                                     f"{t.matrix.dtype} tiles, the task class expects "
                                     f"{d.mb}x{d.nb} {d.dtype}")
                t.matrix.ptr()
                flags[i] = t.flags
                if (d.op | t.flags) & c.PARSEC_AFFINITY:
                    affinity = True
                if ((d.op & c.PARSEC_GET_OP_TYPE) != c.PARSEC_INPUT
                        and not ((d.op | t.flags) & c.PARSEC_DONT_TRACK)):
                    write_flows += 1
            else:
                p.store(args[i], &values[i])
                ptrs[i] = &values[i]
                flags[i] = 0
        if ctx.nb_nodes > 1 and write_flows >= 1 and not affinity:
            raise ValueError(f"task class {tc.name} writes a tile but no parameter has AFFINITY: "
                             "a task that writes data must be placed in a distributed run")

        for i in range(nargs):
            if isinstance(tc.params[i], Data):
                t = <Tile>args[i]
                ptrs[i] = self.dtd_tile(t)
                self._collections[t.matrix] = True

        ctc = tc._tc
        with nogil:
            c.pyparsec_dtd_insert(tp, ctc, priority, device, nargs, flags, ptrs)
        self._needs_wait = True

    cdef c.parsec_dtd_tile_t *dtd_tile(self, Tile t):
        t.matrix.dtd_register()
        self._collections.setdefault(t.matrix, False)
        return c.parsec_dtd_tile_of(t.matrix.dc(), t.key)

    def data_flush(self, Tile tile not None):
        """Flush one tile back to its owner (``parsec_dtd_data_flush``)."""
        cdef c.parsec_taskpool_t *tp = self._tp
        cdef c.parsec_dtd_tile_t *dtile
        self.attached_context()
        dtile = self.dtd_tile(tile)
        with nogil:
            c.parsec_dtd_data_flush(tp, dtile)
        self._needs_wait = True

    def data_flush_all(self, TiledMatrix matrix not None):
        """Flush every tile of ``matrix`` back to its owner
        (``parsec_dtd_data_flush_all``); :meth:`wait` does it for the
        collections that need it."""
        cdef c.parsec_taskpool_t *tp = self._tp
        cdef c.parsec_data_collection_t *dc
        self.attached_context()
        matrix.dtd_register()
        dc = matrix.dc()
        with nogil:
            c.parsec_dtd_data_flush_all(tp, dc)
        self._collections[matrix] = False
        self._needs_wait = True

    cdef check_errors(self):
        cdef TaskClass tc
        for tc in self._classes:
            if tc._error is not None:
                error, tc._error = tc._error, None
                raise error

    cdef flush_collections(self):
        cdef TiledMatrix matrix
        for matrix, used in list(self._collections.items()):
            if used:
                self.data_flush_all(matrix)

    cdef destroy(self):
        # A DTD taskpool can be freed whether or not it was enqueued.
        cdef TaskClass tc
        for tc in self._classes:
            tc.release(self._tp)
        self._classes = []
        self._destruct(self._tp)
        BaseTaskpool.forget(self)
        # The collections are kept alive until no task can refer to them.
        self._collections = {}
