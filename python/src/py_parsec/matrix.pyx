# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Tiled matrices (``parsec_tiled_matrix_t``) and the library operations on them.

The data collections defined here are not tied to a DSL: the same matrix can
be used by DTD tasks (:mod:`py_parsec.dtd`) and by PTG taskpools such as
:func:`redistribute_new`. Tiles are stored in tile (column-major per tile)
layout; :meth:`TiledMatrix.tile` exposes a local tile as a writable
``(mb, nb)`` Fortran-ordered NumPy array that shares the PaRSEC storage.
"""

cimport cython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset

import numpy as np
cimport numpy as cnp

from py_parsec cimport cparsec as c
from py_parsec.runtime cimport Context, Taskpool

cnp.import_array()

_MATRIX_TYPES = {
    np.dtype(np.int8): c.PARSEC_MATRIX_BYTE,
    np.dtype(np.uint8): c.PARSEC_MATRIX_BYTE,
    np.dtype(np.int32): c.PARSEC_MATRIX_INTEGER,
    np.dtype(np.float32): c.PARSEC_MATRIX_FLOAT,
    np.dtype(np.float64): c.PARSEC_MATRIX_DOUBLE,
    np.dtype(np.complex64): c.PARSEC_MATRIX_COMPLEX_FLOAT,
    np.dtype(np.complex128): c.PARSEC_MATRIX_COMPLEX_DOUBLE,
}


def matrix_type(dtype):
    """The ``parsec_matrix_type_t`` matching a NumPy dtype."""
    dt = np.dtype(dtype)
    try:
        return _MATRIX_TYPES[dt]
    except KeyError:
        raise TypeError(f"no PaRSEC matrix type for dtype {dt}") from None


cdef object host_array(void *ptr, int mb, int nb, int typenum, object owner):
    """An (mb, nb) column-major array over ptr; it keeps owner alive."""
    cdef cnp.npy_intp dims[2]
    cdef cnp.ndarray arr
    dims[0] = nb
    dims[1] = mb
    arr = cnp.PyArray_SimpleNewFromData(2, dims, typenum, ptr)
    if owner is not None:
        cnp.set_array_base(arr, owner)
    return arr.T


def _default_grid(int nodes):
    """Near-square P x Q process grid with P <= Q."""
    cdef int p = <int>(nodes ** 0.5)
    while p > 1 and nodes % p:
        p -= 1
    return max(p, 1)


@cython.no_gc_clear   # keeps _context until the matrix is released (see runtime.Taskpool)
cdef class TiledMatrix:
    """Base class of the tiled matrices; see :class:`BlockCyclic`."""

    def __cinit__(self, *args, **kwargs):
        self._desc = NULL
        self._context = None
        self._dtd_registered = False

    def __dealloc__(self):
        # The matrix holds a reference to its context, so the context is
        # still initialized here unless fini() already released the matrix.
        if self._desc != NULL:
            self.dtd_unregister()
            self.destroy()
            self._desc = NULL

    cdef parsec_tiled_matrix_t *ptr(self) except NULL:
        if self._desc == NULL:
            raise RuntimeError("the matrix has been released")
        return self._desc

    cdef parsec_data_collection_t *dc(self) except NULL:
        return &self.ptr().super

    cdef dtd_register(self):
        """Give the collection the DTD tile table; done on first DTD use."""
        if not self._dtd_registered:
            c.parsec_dtd_data_collection_init(self.dc())
            self._dtd_registered = True

    cdef dtd_unregister(self):
        if self._dtd_registered:
            c.parsec_dtd_data_collection_fini(self.dc())
            self._dtd_registered = False

    cdef destroy(self):
        c.parsec_tiled_matrix_destroy(self._desc)

    def release(self):
        """Free the matrix storage; tile arrays obtained earlier become invalid."""
        if self._desc == NULL:
            return
        self.dtd_unregister()
        self.destroy()
        self._desc = NULL

    @property
    def address(self):
        """The ``parsec_tiled_matrix_t *`` as an int, for C libraries."""
        return <uintptr_t>self.ptr()

    @property
    def name(self):
        return self._name.decode()

    @property
    def m(self):
        """Number of rows."""
        return self.ptr().m

    @property
    def n(self):
        """Number of columns."""
        return self.ptr().n

    @property
    def mb(self):
        """Number of rows of a tile."""
        return self.ptr().mb

    @property
    def nb(self):
        """Number of columns of a tile."""
        return self.ptr().nb

    @property
    def mt(self):
        """Number of tile rows."""
        return self.ptr().mt

    @property
    def nt(self):
        """Number of tile columns."""
        return self.ptr().nt

    @property
    def nb_local_tiles(self):
        return self.ptr().nb_local_tiles

    @property
    def rank(self):
        """Rank of this process in the collection."""
        return self.ptr().super.myrank

    @property
    def nodes(self):
        """Number of processes the collection is distributed over."""
        return self.ptr().super.nodes

    cdef int _check_tile(self, int m, int n) except -1:
        cdef parsec_tiled_matrix_t *desc = self.ptr()
        if not (0 <= m < desc.mt and 0 <= n < desc.nt):
            raise IndexError(f"tile ({m}, {n}) outside the {desc.mt}x{desc.nt} tiles of {self.name}")
        return 0

    def rank_of(self, int m, int n):
        """Rank owning tile (m, n)."""
        self._check_tile(m, n)
        return c.pyparsec_rank_of(self.dc(), m, n)

    def is_local(self, int m, int n):
        """True if tile (m, n) is stored by this process."""
        return self.rank_of(m, n) == self.rank

    def tile(self, int m, int n):
        """Local tile (m, n) as an (mb, nb) Fortran-ordered array (no copy).

        The array refers to the host copy of the tile: read it once the
        taskpools writing the tile have completed (and the data has been
        flushed back for DTD).
        """
        cdef void *ptr
        self._check_tile(m, n)
        ptr = c.pyparsec_local_tile(self.dc(), m, n)
        if ptr == NULL:
            raise ValueError(f"tile ({m}, {n}) of {self.name} is owned by rank "
                             f"{self.rank_of(m, n)}, not by rank {self.rank}")
        return host_array(ptr, self.ptr().mb, self.ptr().nb, self._typenum, self)

    def local_tiles(self):
        """Iterate over (m, n, tile array) for the tiles stored by this process."""
        cdef int m, n
        cdef parsec_tiled_matrix_t *desc = self.ptr()
        for m in range(desc.mt):
            for n in range(desc.nt):
                if c.pyparsec_rank_of(&desc.super, m, n) == <int>desc.super.myrank:
                    yield m, n, self.tile(m, n)

    def __repr__(self):
        if self._desc == NULL:
            return f"<py_parsec.{type(self).__name__} released>"
        return (f"<py_parsec.{type(self).__name__} {self.name!r} {self.m}x{self.n} "
                f"tiles {self.mb}x{self.nb} {self.dtype}>")


cdef class BlockCyclic(TiledMatrix):
    """Two-dimensional block-cyclic tiled matrix (``parsec_matrix_block_cyclic_t``).

    Parameters
    ----------
    context : Context
    m, n : int
        Matrix size.
    mb, nb : int
        Tile size.
    dtype : numpy dtype
        float32, float64, complex64, complex128, int32, int8 or uint8.
    P : int, optional
        Rows of the process grid; Q is ``context.nb_nodes // P``. The
        default is a near-square grid.
    kp, kq, ip, jq : int
        k-cyclicity and grid origin, as in ``parsec_matrix_block_cyclic_init``.
    name : str
        Key of the data collection (shown in traces).

    The local tiles are allocated and zeroed.
    """

    def __cinit__(self, Context context not None, int m, int n, int mb, int nb,
                  dtype=np.float64, P=None, int kp=1, int kq=1, int ip=0, int jq=0,
                  name="matrix"):
        cdef int mtype, p, q, nodes = context.nb_nodes
        cdef size_t nbytes

        self._bc = NULL
        if m < 0 or n < 0 or mb <= 0 or nb <= 0:
            raise ValueError("matrix and tile sizes must be positive")
        p = _default_grid(nodes) if P is None else P
        if p <= 0 or nodes % p:
            raise ValueError(f"P={p} does not divide the {nodes} processes")
        q = nodes // p
        self.dtype = np.dtype(dtype)
        self._typenum = self.dtype.num
        mtype = matrix_type(self.dtype)
        self._name = name.encode()

        self._bc = <c.parsec_matrix_block_cyclic_t *>calloc(1, sizeof(c.parsec_matrix_block_cyclic_t))
        if self._bc == NULL:
            raise MemoryError()
        c.parsec_matrix_block_cyclic_init(self._bc, <c.parsec_matrix_type_t>mtype, c.PARSEC_MATRIX_TILE,
                                          context.rank, mb, nb, m, n, 0, 0, m, n,
                                          p, q, kp, kq, ip, jq)
        self._desc = &self._bc.super
        c.parsec_data_collection_set_key(&self._desc.super, self._name)

        nbytes = <size_t>self._desc.nb_local_tiles * self._desc.bsiz * c.pyparsec_sizeof_matrix_type(mtype)
        if nbytes > 0:
            self._bc.mat = c.parsec_data_allocate(nbytes)
            if self._bc.mat == NULL:
                self.release()
                raise MemoryError(f"cannot allocate {nbytes} bytes for {name}")
            memset(self._bc.mat, 0, nbytes)

        self._context = context
        context.register_collection(self)

    cdef destroy(self):
        if self._bc.mat != NULL:
            c.parsec_data_free(self._bc.mat)
            self._bc.mat = NULL
        TiledMatrix.destroy(self)
        free(self._bc)
        self._bc = NULL


def redistribute_new(TiledMatrix source not None, TiledMatrix target not None,
                     size_row=None, size_col=None,
                     int disi_source=0, int disj_source=0,
                     int disi_target=0, int disj_target=0):
    """PTG taskpool copying a size_row x size_col submatrix of source into target.

    The submatrix starts at (disi_source, disj_source) in source and lands at
    (disi_target, disj_target) in target; the two matrices may use any
    distribution. Add the returned taskpool to a context and wait for it like
    any other taskpool. The size defaults to the whole source matrix.
    """
    cdef c.parsec_taskpool_t *tp
    cdef Taskpool wrapper
    cdef int rows = source.m - disi_source if size_row is None else size_row
    cdef int cols = source.n - disj_source if size_col is None else size_col

    if source.dtype != target.dtype:
        raise TypeError(f"source and target dtypes differ ({source.dtype} vs {target.dtype})")
    tp = c.parsec_redistribute_New(source.ptr(), target.ptr(), rows, cols,
                                   disi_source, disj_source, disi_target, disj_target)
    if tp == NULL:
        raise ValueError("parsec_redistribute_New rejected the arguments")
    wrapper = Taskpool.from_address(<uintptr_t>tp, keepalive=(source, target))
    return wrapper


def redistribute_dtd(Context context not None, TiledMatrix source not None, TiledMatrix target not None,
                     size_row=None, size_col=None,
                     int disi_source=0, int disj_source=0,
                     int disi_target=0, int disj_target=0):
    """Blocking DTD version of :func:`redistribute_new`.

    It starts the context and returns after ``parsec_context_wait``: every
    taskpool of the context has then completed and the context is idle.
    Pending DTD work on source and target must have been flushed and waited.
    """
    cdef int rc
    cdef int rows = source.m - disi_source if size_row is None else size_row
    cdef int cols = source.n - disj_source if size_col is None else size_col
    cdef c.parsec_context_t *ctx = context.ptr()
    cdef c.parsec_tiled_matrix_t *src = source.ptr()
    cdef c.parsec_tiled_matrix_t *dst = target.ptr()

    if source.dtype != target.dtype:
        raise TypeError(f"source and target dtypes differ ({source.dtype} vs {target.dtype})")
    # parsec_redistribute_dtd() sets up and tears down the DTD tile tables of
    # both collections itself.
    source.dtd_unregister()
    target.dtd_unregister()
    with nogil:
        rc = c.parsec_redistribute_dtd(ctx, src, dst, rows, cols,
                                       disi_source, disj_source, disi_target, disj_target)
    if rc != c.PARSEC_SUCCESS:
        raise RuntimeError(f"parsec_redistribute_dtd failed ({rc})")
