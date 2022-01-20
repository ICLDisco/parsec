This file contains the main features as well as overviews of specific
changes to this project (since v1.1.0).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


Unreleased (master)
-------------------


### Added

 - Add DTD CUDA support including NEW tiles in DTD

 - PaRSEC API 4.0 (still changing)

 - Enable users to manage their own data copies without PaRSEC
   interfering. Data copies are marked as being owned by PaRSEC or
   not and managed by PaRSEC or not. A data copy owned by PaRSEC can
   be reclaimed by PaRSEC when its reference count reaches 0, a data
   copy managed by PaRSEC can be copied / moved onto a different
   device, while a data copy not managed by PaRSEC will never be
   moved by the runtime.

 - Add an info system, and introduce two info hooks. See parsec/class/info.h
   for details. The info system allows the user to register info objects
   with different levels of structures and dynamic objects in the PaRSEC
   runtime.

 - PTG supports user-defined routines to move data between GPU and
   CPU, and user-defined sizes for buffers allocated on the GPU.

 - PTG supports reshaping data propagated between local tasks and
   the speficiation of two types on acccesses to data colletions.

### Changed
 
 - Renamed symbols related to data distribution to properly prefix them with
   the `parsec_` prefix. The old symbols have been deprecated.
 
 - DTD interface change: the global array parsec_dtd_arena_datatypes
   is replaced with functions to create, destroy, and get arena
   datatypes for DTD, and these objects now live inside the
   parsec context.
 
 - PARSEC_SUCCESS changed to 0 (from -1), all values for PARSEC_ERR_XYZ changed.

 - PaRSEC now requires CMake 3.18.

 - PaRSEC profiling system does not require for local dicitonaries to
   be identical between ranks anymore.

### Deprecated
 
 - PaRSEC API 3.0

 - data distribution w/o the `parsec_` prefix. Further documentation (including a
   sed script) can be found in contrib/renaming.

### Removed

 - Removed obsolete dbp2paje tool; h5totrace is the replacement tool
   to use. This removes the optional dependency on GTG.

### Fixed

### Security


v3.0.2012
---------

 - PaRSEC API 3.0

 - PaRSEC now requires CMake 3.16.

 - New configure system to ease the installation of PaRSEC. See
   INSTALL for details. This system automates installation on most DOE
   leadership systems.

 - Split DPLASMA and PaRSEC into separate repositories. PaRSEC moves from
   cmake-2.0 to cmake-3.12, using targets. Targets are exported for
   third-party integration

 - Add visualization tools to extract user-defined properties from the
   application (see: PR 229 visualization-tools)

 - Automate expression of required data transfers from host-to-device and
   device-to-host to satisfy depencencies (and anti-dependencies). PaRSEC tracks
   multiple versions of the same data as data copies with a coherency algorithm
   that initiates data transfers as needed. The heurisitic for the eviction policy
   in out-of-memory event on GPU has been optimized to allow for efficient
   operation in larger than GPU memory problems.

 - Add support for MPI out-of-order matching capabilities; Added capability
   for compute threads to send direct control messages to indicate completion
   of tasks to remote nodes (without delegation to the communication thread)

 - Remove communication mode EAGER from the runtime. It had a rare
   but hard to correct bug that would rarely deadlock, and the performance
   benefit was small.

 - Add a Map operator on the Block Cyclic matrix data collection that
   performs in-place data transformation on the collection with a user provided
   operator.

 - Add support in the runtime for user-defined properties evaluated at
   runtime and easy to export through a shared memory region (see: PR
   229 visualization-tools)

 - Add a PAPI-SDE interface to the parsec library, to expose internal
   counters via the PAPI-Software Defined Events interface.

 - Add a backend support for OTF2 in the profiling mechanism. OTF2 is
   used automatically if a OTF2 installation is found.

 - Add a MCA parameter to control the number of ejected blocks from GPU
   memory (device_cuda_max_number_of_ejected_data). Add a MCA parameter
   to control wether or not the GPU engine will take some time to sort
   the first N tasks of the pending queue (device_cuda_sort_pending_list).

 - Reshape the users vision of PaRSEC: they only have to include a single
   header (parsec.h) for most usages, and link with a single library
   (-lparsec).

 - Update the PaRSEC DSL handling of initial tasks. We now rely on 2
   pieces of information: the number of DSL tasks, and the number of
   tasks imposed by the system (all types of data transfer).

 - Add a purely local scheduler (ll), that uses a single LIFO per
   thread. Each schedule operation does 1 atomic (push in local queue),
   each select operation does up to t atomics (pop in local queue, then
   try any other thread's queue until they are all tested empty).

 - Add a --ignore-properties=... option to parsec_ptgpp

 - Change API of hash tables: allow keys of arbitrary size. The API
   features how to build a key from a task; how to hash a key into
   1 <= N <= 64 bits; and how to compare twy keys (plus a printing
   function to debug).

 - Change behavior of DEBUG_HISTORY: log all information inside
   a buffer of fixed size (MCA parameter) per thread, do not allocate
   memory during logging, and use timestamp to re-order output
   when the user calls dump()

 - DTD interface is updated (new flag to send pointer as parameter,
   unpacking of paramteres is simpler etc).

 - DTD provides mca param (dtd_debug_verbose) to print information
   about traversal of DAG in a separate output stream from the default.


v2.0.0rc2
---------

 - Rename all functions, files, directories from dague/DAGUE/DAGuE to
   parsec/PARSEC/PaRSEC.


v2.0.0rc1
---------

 - .so support. Dynamic Library Build has been succesfully tested on
   Linux platforms, reducing significantly the size of the compiled
   dplasma/testing directory. Note that on modern architectures,
   all depending libraries must be compiled either as Position Independent
   Code (-fPIC) or as shared libraries. Hint: add --cflags="-fPIC" when
   running the plasma-installer.
 - The "inline_c %{ ... %}" block syntax has been simplified to either
   "%c{ ... %}" or "%{ ... %}". The change is backward compatible.


v1.2.0
------

 - The support for C is required from MPI.
 - Revert to an older LEX syntax (not (?i:xxx))
 - Don't recursively call the MPI communication engine progress function.
 - Protect the runtime headers from inclusion in C++ files.
 - Fix a memory leak allowing the remote data to be released once used.
 - Integrate the new profiling system and the python interface (added
   Panda support).
 - Change several default parameters for the DPLASMA tests.
 - Add Fortran support for the PaRSEC and the profiling API and add tests
   to validate it.
 - Backport all the profiling features from the devel branch (panda support,
   simpler management, better integration with python, support for HDF5...).
 - Change the colorscheme of the .dot generator
 - Correctly compute the identifier of the tasks (ticket #33).
 - Allocate the list items from the corresponding list using the requested
   gap.
 - Add support for older lex (without support for ?i).
 - Add support for 128 bits atomics and resolve the lingering ABA issue.
   When 128 bits atomics are not available fall back to an atomic lock
   implementation of the most basic data structures.
 - Required Cython 0.19.1 (at least)
 - Completely disconnect the ordering of parameters and locals in the JDF.
 - Many other minor bug fixes, code readability impeovement and typos.
 - DPLASMA:
   - Add the doxygen documentation generation.
   - Improved ztrmm with all the matrix reads unified.
   - Support for potri functions (trtri+lauum), and corresponding testings.
   - Fix bug in symmetric/hermitian norms when A.mt < P.
