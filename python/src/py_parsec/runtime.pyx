# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""PaRSEC contexts and taskpools.

A :class:`Context` owns one PaRSEC runtime instance (``parsec_init`` /
``parsec_fini``). Every domain specific language produces the same kind of
object, a ``parsec_taskpool_t``, so all of them are wrapped by
:class:`Taskpool` and scheduled on a context the same way, with
:meth:`Context.add_taskpool`. DTD taskpools (:class:`py_parsec.dtd.Taskpool`)
and the PTG taskpools returned by library calls are interchangeable there.
"""

import atexit
import traceback
import warnings
import weakref

cimport cython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from py_parsec cimport cparsec as c

DEV_CPU = c.PARSEC_DEV_CPU
DEV_CUDA = c.PARSEC_DEV_CUDA
DEV_HIP = c.PARSEC_DEV_HIP
DEV_LEVEL_ZERO = c.PARSEC_DEV_LEVEL_ZERO
DEV_ALL = c.PARSEC_DEV_ALL

HOOK_RETURN_DONE = c.PARSEC_HOOK_RETURN_DONE
HOOK_RETURN_AGAIN = c.PARSEC_HOOK_RETURN_AGAIN
HOOK_RETURN_NEXT = c.PARSEC_HOOK_RETURN_NEXT
HOOK_RETURN_DISABLE = c.PARSEC_HOOK_RETURN_DISABLE
HOOK_RETURN_ASYNC = c.PARSEC_HOOK_RETURN_ASYNC
HOOK_RETURN_ERROR = c.PARSEC_HOOK_RETURN_ERROR


def version():
    """Return the (major, minor, release) version of the PaRSEC library."""
    cdef int major = 0, minor = 0, release = 0
    c.parsec_version(&major, &minor, &release)
    return (major, minor, release)


def address_of(obj):
    """Return the C pointer held by a PyCapsule or an int as an int."""
    if isinstance(obj, int):
        return obj
    if PyCapsule_IsValid(obj, NULL):
        return <uintptr_t>PyCapsule_GetPointer(obj, NULL)
    raise TypeError(f"expected an address (int) or an unnamed PyCapsule, got {type(obj).__name__}")


# Contexts still alive when the interpreter exits are finalized before any
# Py_AtExit handler runs (mpi4py finalizes MPI from one of those).
_live_contexts = weakref.WeakSet()
# Number of contexts between parsec_init and parsec_fini. A plain counter:
# the garbage collector clears weak references to unreachable objects before
# it runs any destructor, so the WeakSet cannot be consulted from __dealloc__.
_nb_live_contexts = 0


def runtime_initialized():
    """True while at least one :class:`Context` is initialized."""
    return _nb_live_contexts > 0
# Every taskpool object not yet released, so that a context can free the
# ones never enqueued anywhere before PaRSEC is finalized.
_live_taskpools = weakref.WeakSet()


@atexit.register
def _fini_live_contexts():
    for ctx in list(_live_contexts):
        ctx.fini()



# Neither Context, Taskpool nor TiledMatrix is ever the link that closes a
# reference cycle (a context holds only weak references; taskpools and
# matrices refer to their context one way), so the garbage collector must
# not clear their attributes: taskpools and matrices need their context
# until they are released.
@cython.no_gc_clear
cdef class Context:
    """A PaRSEC runtime instance.

    Parameters
    ----------
    nb_cores : int
        Number of compute threads; a value <= 0 uses every core available to
        the process.
    args : sequence of str, optional
        PaRSEC options, as accepted on the command line after ``--``
        (for instance ``["--mca", "runtime_comm_short_limit", "0"]``).

    PaRSEC initializes MPI if the application did not, and then finalizes it
    in :meth:`fini`; as MPI cannot be initialized twice, import ``mpi4py``
    first to create more than one context in a process. The context can be
    used as a context manager, and is finalized at interpreter exit if still
    alive. Finalizing a context completes and releases the taskpools and data
    collections created on it.
    """

    def __cinit__(self, int nb_cores=-1, args=None):
        cdef int argc
        cdef char **pargv
        cdef Py_ssize_t i

        self._ctx = NULL
        self._c_argv = NULL
        self._dtd_arenas = {}
        self._taskpools = weakref.WeakSet()
        self._collections = weakref.WeakSet()
        self._argv = [a.encode() if isinstance(a, str) else bytes(a) for a in (args or ())]

        argc = <int>len(self._argv)
        self._c_argv = <char **>malloc((argc + 1) * sizeof(char *))
        if self._c_argv == NULL:
            raise MemoryError()
        for i in range(argc):
            self._c_argv[i] = <char *>self._argv[i]
        self._c_argv[argc] = NULL
        pargv = self._c_argv

        with nogil:
            self._ctx = c.parsec_init(nb_cores, &argc, &pargv)
        if self._ctx == NULL:
            raise RuntimeError("parsec_init failed")

        self.rank = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_RANK)
        self.nb_nodes = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_NODES)
        self.nb_cores = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_CORES)
        _live_contexts.add(self)
        global _nb_live_contexts
        _nb_live_contexts += 1

    def __dealloc__(self):
        # Enqueued taskpools and collections hold a reference to their
        # context, so none is left when the context itself is collected.
        global _nb_live_contexts
        if self._ctx != NULL:
            if c.pyparsec_context_is_active(self._ctx):
                with nogil:
                    c.parsec_context_wait(self._ctx)
            for arena_id in self._dtd_arenas.values():
                c.parsec_dtd_free_arena_datatype(self._ctx, arena_id)
            with nogil:
                c.parsec_fini(&self._ctx)
            _nb_live_contexts -= 1
        free(self._c_argv)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fini()
        return False

    cdef parsec_context_t *ptr(self) except NULL:
        if self._ctx == NULL:
            raise RuntimeError("the PaRSEC context has been finalized")
        return self._ctx

    cdef bint is_active(self):
        return self._ctx != NULL and c.pyparsec_context_is_active(self._ctx)

    @property
    def address(self):
        """The ``parsec_context_t *`` as an int, for C libraries driven from Python."""
        return <uintptr_t>self.ptr()

    @property
    def active(self):
        """True between :meth:`start` and the end of :meth:`wait`."""
        return self.is_active()

    def nb_devices(self, int device_type):
        """Number of devices of ``device_type`` (``DEV_CUDA``, ...) PaRSEC uses."""
        cdef int rc = c.parsec_context_query(self.ptr(), c.PARSEC_CONTEXT_QUERY_DEVICES, device_type)
        return max(rc, 0)

    def start(self):
        """Start the compute threads; returns False if they were already running."""
        return c.parsec_context_start(self.ptr()) == 0

    def wait(self):
        """Complete every taskpool enqueued in the context (``parsec_context_wait``).

        DTD taskpools are completed too, after the same flush as their own
        :meth:`wait`. On return the compute threads are idle; call
        :meth:`start` before scheduling more work. An exception raised by a
        Python task body is raised here.
        """
        cdef parsec_context_t *ctx = self.ptr()
        cdef Taskpool tp
        cdef int rc
        if not c.pyparsec_context_is_active(ctx):
            raise RuntimeError("the context has not been started")
        for tp in list(self._taskpools):
            tp.flush_collections()
        with nogil:
            rc = c.parsec_context_wait(ctx)
        if rc < 0:
            raise RuntimeError(f"parsec_context_wait failed ({rc})")
        errors = []
        for tp in list(self._taskpools):
            tp._needs_wait = False
            try:
                tp.check_errors()
            except BaseException as e:
                errors.append(e)
        if errors:
            raise errors[0]

    def add_taskpool(self, Taskpool tp not None):
        """Enqueue a taskpool, of any DSL, in this context."""
        cdef int rc
        if tp._context is not None:
            raise RuntimeError("the taskpool has already been added to a context")
        rc = c.parsec_context_add_taskpool(self.ptr(), tp.ptr())
        if rc != c.PARSEC_SUCCESS:
            raise RuntimeError(f"parsec_context_add_taskpool failed ({rc})")
        tp._context = self
        tp._needs_wait = True
        self._taskpools.add(tp)

    def fini(self):
        """Complete and release the taskpools and collections of this context,
        then finalize it."""
        cdef Taskpool tp
        if self._ctx == NULL:
            return
        errors = []
        try:
            if any((<Taskpool>tp)._needs_wait for tp in self._taskpools):
                self.start()
            if self.is_active():
                self.wait()
        except BaseException as e:
            errors.append(e)
        for tp in list(self._taskpools):
            try:
                tp.release()
            except BaseException as e:
                errors.append(e)
        if self.is_active():
            with nogil:
                c.parsec_context_wait(self._ctx)
        # Taskpools never enqueued in a context cannot be freed once PaRSEC
        # is finalized: free them now, there is nothing to wait for.
        for tp in list(_live_taskpools):
            if tp._context is None:
                tp.release()
        for dc in list(self._collections):
            dc.release()
        for arena_id in self._dtd_arenas.values():
            c.parsec_dtd_free_arena_datatype(self._ctx, arena_id)
        self._dtd_arenas.clear()
        _live_contexts.discard(self)
        with nogil:
            c.parsec_fini(&self._ctx)
        self._ctx = NULL
        global _nb_live_contexts
        _nb_live_contexts -= 1
        if errors:
            raise errors[0]

    cdef int dtd_tile_arena(self, int mtype, int mb, int nb, int ld) except -1:
        """DTD arena id of a full mb x nb tile, created on first use."""
        key = (mtype, mb, nb, ld)
        arena_id = self._dtd_arenas.get(key)
        if arena_id is None:
            arena_id = c.pyparsec_dtd_tile_arena_new(self.ptr(), mtype, mb, nb, ld)
            if arena_id < 0:
                raise RuntimeError(f"cannot create a {mb}x{nb} arena datatype")
            self._dtd_arenas[key] = arena_id
        return arena_id

    cdef register_collection(self, object collection):
        self._collections.add(collection)

    def __repr__(self):
        if self._ctx == NULL:
            return "<py_parsec.Context finalized>"
        return f"<py_parsec.Context rank {self.rank}/{self.nb_nodes}, {self.nb_cores} cores>"


cdef void _default_destructor(c.parsec_taskpool_t *tp) noexcept nogil:
    c.parsec_taskpool_free(tp)


@cython.no_gc_clear   # see Context
cdef class Taskpool:
    """A PaRSEC taskpool, whatever DSL created it.

    Library calls that build PTG taskpools return instances of this class;
    :meth:`from_address` wraps a ``parsec_taskpool_t *`` created by C code
    (for instance a JDF compiled into a separate shared library), so it can be
    scheduled next to DTD taskpools in the same context.

    A taskpool is freed by :meth:`free`, when it is garbage collected, or by
    the finalization of its context; work still pending at that point is
    completed first, as by :meth:`wait`. PaRSEC cannot release a PTG
    taskpool that was never enqueued (its destructor unregisters it from the
    context), so such a taskpool is leaked, with a ``ResourceWarning``, when
    released; DTD taskpools do not have this limit.
    """

    def __cinit__(self, *args, **kwargs):
        self._tp = NULL
        self._destruct = _default_destructor
        self._context = None
        self._keepalive = []
        self._needs_wait = False
        _live_taskpools.add(self)

    def __dealloc__(self):
        # Runs after the attributes of subclasses are cleared, so it is only
        # effective for instances of this class (subclasses release their
        # taskpool in their own __dealloc__ and leave _tp NULL). A taskpool
        # dropped with work pending is completed first, and leaked rather
        # than destroyed if that fails. One that was never enqueued is leaked
        # (see destroy()), and nothing is freed once its context is finalized.
        if self._tp != NULL and self._context is not None and self._context._ctx != NULL:
            if self._needs_wait:
                try:
                    if not self._context.is_active():
                        self._context.start()
                    Taskpool.wait(self)
                except BaseException:
                    traceback.print_exc()
            if not self._needs_wait:
                self._destruct(self._tp)
        self._tp = NULL

    @staticmethod
    def from_address(address, keepalive=()):
        """Take ownership of a ``parsec_taskpool_t *`` (int or PyCapsule).

        The taskpool is released with ``parsec_taskpool_free()``;
        ``keepalive`` lists Python objects (data collections, ...) that must
        outlive it.
        """
        cdef Taskpool tp = Taskpool.__new__(Taskpool)
        cdef uintptr_t addr = address_of(address)
        if addr == 0:
            raise ValueError("NULL taskpool")
        tp._tp = <c.parsec_taskpool_t *>addr
        tp._keepalive = list(keepalive)
        return tp

    cdef parsec_taskpool_t *ptr(self) except NULL:
        if self._tp == NULL:
            raise RuntimeError("the taskpool has been released")
        return self._tp

    @property
    def address(self):
        """The ``parsec_taskpool_t *`` as an int."""
        return <uintptr_t>self.ptr()

    @property
    def context(self):
        """The context the taskpool was added to, or None."""
        return self._context

    def wait(self):
        """Progress the context until this taskpool completes.

        The taskpool must have been added to a started context.
        """
        cdef c.parsec_taskpool_t *tp = self.ptr()
        cdef int rc
        if self._context is None:
            raise RuntimeError("the taskpool has not been added to a context")
        if not self._context.is_active():
            raise RuntimeError("the context of the taskpool has not been started")
        self.flush_collections()
        with nogil:
            rc = c.parsec_taskpool_wait(tp)
        if rc < 0:
            raise RuntimeError(f"parsec_taskpool_wait failed ({rc})")
        self._needs_wait = False
        self.check_errors()

    def free(self):
        """Release the taskpool, completing it first (starting the context if
        needed) if it was not waited."""
        self.release()

    cdef flush_collections(self):
        """Hand the data back to the collections before a wait (DTD)."""
        pass

    cdef check_errors(self):
        pass

    cdef release(self):
        if self._tp == NULL:
            return
        try:
            if self._needs_wait:
                if not self._context.is_active():
                    self._context.start()
                Taskpool.wait(self)
        finally:
            # A taskpool that could not be waited is kept, the error tells why.
            if not self._needs_wait:
                self.destroy()

    cdef destroy(self):
        if self._context is not None:
            self._destruct(self._tp)
        else:
            warnings.warn(f"{self!r} was never enqueued in a context: PaRSEC cannot release it, "
                          "the taskpool is leaked", ResourceWarning, stacklevel=2)
        self.forget()

    cdef forget(self):
        """Drop the Python side of the taskpool."""
        self._tp = NULL
        if self._context is not None:
            self._context._taskpools.discard(self)
        self._keepalive = []

    def __repr__(self):
        where = "released" if self._tp == NULL else hex(<uintptr_t>self._tp)
        return f"<py_parsec.{type(self).__name__} {where}>"
