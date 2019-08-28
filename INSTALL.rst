=================
INSTALLING PaRSEC
=================

.. contents:: Table of Contents

Software Dependencies
=====================

To compile PaRSEC on a new platform, you will need some of the software
below. From 1 to 2 (included) they are mandatory. Everything else is
optional, they provide nice features not critical to the normal usage
of this software package.

1. cmake version 3.11 or above. cmake can be found in the debian
   package cmake, or as sources at the CMake_ download page
2. Any MPI library Open MPI, MPICH2, MVAPICH or any vendor blessed
   implementation.
3. hwloc_ for processor and memory locality features
4. For using PINS (instrumentation based on PAPI) PAPI_ is required
5. For the profiling tools you need several libraries.

   - GTG_ for the trace generation
   - Vite_ a visualization environment (only required for visualization)
   - GD_ usually available on most of the Linux distribution via GraphViz
     installation

.. _CMake: http://www.cmake.org/
.. _hwloc: http://www.open-mpi.org/projects/hwloc/
.. _PAPI: http://icl.cs.utk.edu/papi/
.. _GTG: https://gforge.inria.fr/projects/gtg/
.. _Vite: https://gforge.inria.fr/projects/vite/
.. _GD: http://www.graphviz.org/

Configuring PaRSEC for a new platform
=====================================

PaRSEC is a CMake_ built project. CMake has a comparable goal to
configure_, but it's subtly different. For one thing, CMake display the
commands with colors, but this is not necessarily its most prominent
feature.

CMake keeps everything it found hitherto in a cache file named
``CMakeCache.txt``. Until you have successfully configured PaRSEC,
remove the ``CMakeCache.txt`` file each time you run ``cmake``.

.. _configure: https://www.gnu.org/software/autoconf/

The configure script
--------------------

Passing options to CMake can be confusing. For that reason we have
designed the ``configure`` script that basically wraps around the
invocation of ``cmake`` with a tested and trusted feel to it.

.. code:: bash

  configure --prefix=$INSTALLDIR --with-mpi=$MPI_DIR --with-cuda --disable-debug

will produce a ``cmake`` command line that matches these options,
and execute it.

You can review what ``cmake`` invocation has been produced by looking
into ``config.log``.

It also produces a ``config.status`` script that helps redo the last
``configure`` step, while ``config.status --recheck`` will also clean
the CMake caches.

Not all options you can pass to PaRSEC exist as a ``--enable-xxx``
``--with-yyy`` configure argument. You can pass environment variables
to the produced ``cmake`` command as well as CMake *defines* (both
will appear in the ``config.log``) by using the following form:

.. code:: bash

    configure CC=icc FC=ftn CXX=icpc -DPARSEC_EAGER_LIMIT=0

Plarform Files
--------------

Platform files, found in ``contrib/platforms`` let us distribute recipes
for well known systems that may be similar to a supercomputer near you.
For example, the ``ibm.ac922.summit`` file is intended to compile on the
eponyme Oak Ridge Leadership Computing Facility system.

.. code:: bash

  configure --prefix=$INSTALL_DIR --with-platform=ibm.ac922.summit --disable-debug

This call should get you running in no time on that machine, and you
may still customize and overide the platform file with command line
arguments.

We also provide a ``macosx`` platform file that helps dealing with the
detection of the Fortran compiler on this architecture.

Of course you may edit and produce your own platform files for your
favorite computer. These are shell script that execute in the context
of the main configure script. For example, our continuous integration
system is named *saturn*, in that script you will find examples of
how one sets some default options.

.. code:: bash

  with_hwloc=${HWLOC_ROOT:="/spack/opt/spack/linux-scientific7-x86_64/gcc-7.3.0/hwloc-1.11.11-nu65xwuyodswr74llx3ymi67hgd6vmwe"}
  with_gtg=${GTG_ROOT:="/sw/gtg/0.2-2"}
  with_omega=${OMEGA_ROOT:="/sw/omega/2.1"}

  # BLAS: use MKL
  [ -z "${MKLROOT}" ] || module load intel-mkl/2019.3.199/gcc-7.3.0-2pn4
  with_blas=Intel10_64lp_seq

  # Slurm test options
  CMAKE_DEFINES+=" -DCTEST_MPI_LAUNCHER=\"srun -Ccauchy -N\" -DCTEST_SHM_LAUNCHER=\"srun -Ccauchy\" -DCTEST_GPU_LAUNCHER_OPTIONS=-Cgtx1060"

As you can see, the platform file may contain commands, shell scripts,
load environment modules_, etc. Of note are the ``CMAKE_DEFINES`` and
``ENVVARS`` variables which control what ``-DX=Y`` options are appended
, and ``A=B`` environment are prepended to the ``cmake`` invocation,
respectively.

Cross Compiling
---------------

On some system, the build machine cannot execute the code produced for
compute nodes. An example is the ANL Theta system, a Cray XC40
with Xeon Phi nodes and Haswell build frontends.

Cross compiling is heavily reliant on the *platform file* feature.
For example, on the Theta system, one can cross compile by simply
calling

.. code:: bash

  configure --with-platform=cray.xc40.theta

In this case, the configuration stage will also include a build stage
to produce some of the utilities needed to compile PaRSEC. After
the configure state has completed, you will find in your build directory
a subdirectory named ``native`` that contains profiling and devellopper
tools that can be used on the frontend system.

After the configure step has completed, the build step is carried out
as usual by simply using ``make``.

If you face a new system where you need to cross compile, a good start
is to copy the ``contrib/platforms/cray.xc40.theta`` file, and
customize it according to your needs.

Note that you will most probably need to produce your own ``toolchain``
CMake cross-compilation file. More information can be found about them
on the cmake-toolchain_ web page.

.. _cmake-toolchain: https://cmake.org/cmake/help/v3.14/manual/cmake-toolchains.7.html?highlight=cross

Legacy Configurations
---------------------

Of course, you can always directly invoke ``cmake``. You can take
inspiration from the command produced from the ``configure`` script,
or you can look at the obsolete ``contrib/platforms/legacy/config.inc``.

.. code:: bash

  rm -f CMakeCache.txt
  cmake . -G 'Unix Makefiles' -DPARSEC_DIST_WITH_MPI=ON

``contrib/platforms/legacy`` also contains shell scripts that we used to
configure on older systems. ``config.jaguar`` is for, you got it, XT5,
etc. If your system is similar to one of these old systems, we advise
you to start from a modern platform file and tweak from there by importing
the content of the old scripts. Unlike modern platform files, legacy
scripts are shell scripts that can be executed directly from desired
build directory (VPATH or not).


Full configuration example
--------------------------

Hopefully, once the expected arguments are provided the output will look similar to

.. code:: console

  -- The C compiler identification is GNU 7.4.0
  -- Checking whether C compiler has -isysroot
  -- Checking whether C compiler has -isysroot - yes
  -- Checking whether C compiler supports OSX deployment target flag
  -- Checking whether C compiler supports OSX deployment target flag - yes
  -- Check for working C compiler: /opt/local/bin/gcc
  -- Check for working C compiler: /opt/local/bin/gcc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- The Fortran compiler identification is GNU 7.4.0
  -- Checking whether Fortran compiler has -isysroot
  -- Checking whether Fortran compiler has -isysroot - yes
  -- Checking whether Fortran compiler supports OSX deployment target flag
  -- Checking whether Fortran compiler supports OSX deployment target flag - yes
  -- Check for working Fortran compiler: /opt/local/bin/gfortran
  -- Check for working Fortran compiler: /opt/local/bin/gfortran  -- works
  -- Detecting Fortran compiler ABI info
  -- Detecting Fortran compiler ABI info - done
  -- Checking whether /opt/local/bin/gfortran supports Fortran 90
  -- Checking whether /opt/local/bin/gfortran supports Fortran 90 -- yes
  -- The CXX compiler identification is GNU 7.4.0
  -- Checking whether CXX compiler has -isysroot
  -- Checking whether CXX compiler has -isysroot - yes
  -- Checking whether CXX compiler supports OSX deployment target flag
  -- Checking whether CXX compiler supports OSX deployment target flag - yes
  -- Check for working CXX compiler: /opt/local/bin/g++
  -- Check for working CXX compiler: /opt/local/bin/g++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Found BISON: /opt/local/bin/bison (found version "3.4.1")
  -- Found FLEX: /opt/local/bin/flex (found version "2.6.4")
  -- Building for target x86_64
  -- Found target X86_64
  -- Performing Test C_M32or64
  -- Performing Test C_M32or64 - Success
  -- Performing Test PARSEC_HAVE_STD_C1x
  -- Performing Test PARSEC_HAVE_STD_C1x - Success
  -- Performing Test PARSEC_HAVE_WALL
  -- Performing Test PARSEC_HAVE_WALL - Success
  -- Performing Test PARSEC_HAVE_WEXTRA
  -- Performing Test PARSEC_HAVE_WEXTRA - Success
  -- Performing Test PARSEC_HAVE_PAR_EQUALITY
  -- Performing Test PARSEC_HAVE_PAR_EQUALITY - Success
  -- Performing Test PARSEC_HAVE_G3
  -- Performing Test PARSEC_HAVE_G3 - Success
  -- Looking for sys/types.h
  -- Looking for sys/types.h - found
  -- Looking for stdint.h
  -- Looking for stdint.h - found
  -- Looking for stddef.h
  -- Looking for stddef.h - found
  -- Check size of __int128_t
  -- Check size of __int128_t - done
  -- Performing Test PARSEC_COMPILER_C11_COMPLIANT
  -- Performing Test PARSEC_COMPILER_C11_COMPLIANT - Success
  -- Performing Test PARSEC_STDC_HAVE_C11_ATOMICS
  -- Performing Test PARSEC_STDC_HAVE_C11_ATOMICS - Success
  -- Looking for include file stdatomic.h
  -- Looking for include file stdatomic.h - found
  -- Performing Test PARSEC_ATOMIC_USE_C11_32
  -- Performing Test PARSEC_ATOMIC_USE_C11_32 - Success
  -- Performing Test PARSEC_ATOMIC_USE_C11_64
  -- Performing Test PARSEC_ATOMIC_USE_C11_64 - Success
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Failed
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Failed
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Success
  -- 	 support for 32 bits atomics - found
  -- 	 support for 64 bits atomics - found
  -- 	 support for 128 bits atomics - found
  -- Looking for pthread.h
  -- Looking for pthread.h - found
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
  -- Found Threads: TRUE
  -- Looking for pthread_getspecific
  -- Looking for pthread_getspecific - found
  -- Looking for pthread_barrier_init
  -- Looking for pthread_barrier_init - not found
  -- Looking for pthread_barrier_init
  -- Looking for pthread_barrier_init - not found
  -- Looking for sched_setaffinity
  -- Looking for sched_setaffinity - not found
  -- Looking for sched_setaffinity in rt
  -- Looking for sched_setaffinity in rt - not found
  -- Performing Test PARSEC_HAVE_TIMESPEC_TV_NSEC
  -- Performing Test PARSEC_HAVE_TIMESPEC_TV_NSEC - Success
  -- Looking for clock_gettime in c
  -- Looking for clock_gettime in c - found
  -- Looking for include file stdarg.h
  -- Looking for include file stdarg.h - found
  -- Performing Test PARSEC_HAVE_VA_COPY
  -- Performing Test PARSEC_HAVE_VA_COPY - Success
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_FORMAT_PRINTF
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_FORMAT_PRINTF - Success
  -- Performing Test PARSEC_HAVE_THREAD_LOCAL
  -- Performing Test PARSEC_HAVE_THREAD_LOCAL - Success
  -- Looking for asprintf
  -- Looking for asprintf - found
  -- Looking for vasprintf
  -- Looking for vasprintf - found
  -- Looking for include file unistd.h
  -- Looking for include file unistd.h - found
  -- Looking for include file getopt.h
  -- Looking for include file getopt.h - found
  -- Looking for getopt_long
  -- Looking for getopt_long - found
  -- Looking for include file errno.h
  -- Looking for include file errno.h - found
  -- Looking for include file stddef.h
  -- Looking for include file stddef.h - found
  -- Looking for include file stdbool.h
  -- Looking for include file stdbool.h - found
  -- Looking for include file ctype.h
  -- Looking for include file ctype.h - found
  -- Performing Test PARSEC_HAVE_BUILTIN_CPU
  -- Performing Test PARSEC_HAVE_BUILTIN_CPU - Success
  -- Performing Test PARSEC_HAVE_BUILTIN_CPU512
  -- Performing Test PARSEC_HAVE_BUILTIN_CPU512 - Success
  -- Looking for getrusage
  -- Looking for getrusage - found
  -- Looking for RUSAGE_THREAD
  -- Looking for RUSAGE_THREAD - not found
  -- Looking for RUSAGE_THREAD
  -- Looking for RUSAGE_THREAD - not found
  -- Looking for include file limits.h
  -- Looking for include file limits.h - found
  -- Looking for include file string.h
  -- Looking for include file string.h - found
  -- Looking for include file libgen.h
  -- Looking for include file libgen.h - found
  -- Looking for include file complex.h
  -- Looking for include file complex.h - found
  -- Looking for include file sys/param.h
  -- Looking for include file sys/param.h - found
  -- Looking for include file sys/types.h
  -- Looking for include file sys/types.h - found
  -- Looking for include file syslog.h
  -- Looking for include file syslog.h - found
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_ALWAYS_INLINE
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_ALWAYS_INLINE - Success
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_VISIBILITY
  -- Performing Test PARSEC_HAVE_ATTRIBUTE_VISIBILITY - Success
  -- Performing Test PARSEC_HAVE_BUILTIN_EXPECT
  -- Performing Test PARSEC_HAVE_BUILTIN_EXPECT - Success
  -- Looking for dlsym
  -- Looking for dlsym - found
  -- Found HWLOC: /opt/local/lib/libhwloc.dylib
  -- Performing Test PARSEC_HAVE_HWLOC_PARENT_MEMBER
  -- Performing Test PARSEC_HAVE_HWLOC_PARENT_MEMBER - Success
  -- Performing Test PARSEC_HAVE_HWLOC_CACHE_ATTR
  -- Performing Test PARSEC_HAVE_HWLOC_CACHE_ATTR - Success
  -- Performing Test PARSEC_HAVE_HWLOC_OBJ_PU
  -- Performing Test PARSEC_HAVE_HWLOC_OBJ_PU - Success
  -- Looking for hwloc_bitmap_free in /opt/local/lib/libhwloc.dylib
  -- Looking for hwloc_bitmap_free in /opt/local/lib/libhwloc.dylib - found
  -- Performing Test MPI_WORKS_WITH_WRAPPER
  -- Performing Test MPI_WORKS_WITH_WRAPPER - Failed
  -- Found MPI_C: /opt/ompi/master/debug/lib/libmpi.dylib (found version "3.1")
  -- Found MPI_CXX: /opt/ompi/master/debug/lib/libmpi_cxx.dylib (found version "3.1")
  -- Found MPI_Fortran: /opt/ompi/master/debug/lib/libmpi_usempif08.dylib (found version "3.1")
  -- Found MPI: TRUE (found version "3.1")
  -- Looking for MPI_Type_create_resized
  -- Looking for MPI_Type_create_resized - found
  -- Performing Test PARSEC_HAVE_MPI_OVERTAKE
  -- Performing Test PARSEC_HAVE_MPI_OVERTAKE - Success
  -- Looking for include file Ayudame.h
  -- Looking for include file Ayudame.h - not found
  -- Fortran adds libraries path /opt/local/lib/gcc7/gcc/x86_64-apple-darwin18/7.4.0;/opt/local/lib/gcc7;/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/usr/lib
  -- Fortran adds libraries gfortran;gcc_ext.10.5;gcc;quadmath;m;gcc_ext.10.5;gcc
  -- Found GTG: /opt/lib/libgtg.dylib
  -- Checking for module 'libgvc'
  --   Found libgvc, version 2.40.1
  -- Found GRAPHVIZ: /opt/local/lib/libgvc.dylib;/opt/local/lib/libcgraph.dylib;/opt/local/lib/libcdt.dylib;/opt/local/lib/libpathplan.dylib
  -- Looking for gdImagePng in /opt/local/lib/libgd.dylib
  -- Looking for gdImagePng in /opt/local/lib/libgd.dylib - found
  -- Found ZLIB: /opt/local/lib/libz.dylib (found version "1.2.11")
  -- Found PNG: /opt/local/lib/libpng.dylib (found version "1.4.12")
  -- Looking for gdImageJpeg in /opt/local/lib/libgd.dylib
  -- Looking for gdImageJpeg in /opt/local/lib/libgd.dylib - found
  -- Found JPEG: /opt/local/lib/libjpeg.dylib (found version "80")
  -- Looking for gdImageGif in /opt/local/lib/libgd.dylib
  -- Looking for gdImageGif in /opt/local/lib/libgd.dylib - found
  -- Found PythonInterp: /opt/local/bin/python (found version "2.7.16")
  -- Cython version 0.29.13 found
  -- Found Cython: /opt/local/bin/cython (Required is at least version "0.21.2")
  -- Looking for shm_open
  -- Looking for shm_open - found
  -- PARSEC Modular Component Architecture (MCA) discovery:
  -- -- Found Component `pins'
  -- Module alperf not selectable: PARSEC_PROF_TRACE disabled.
  -- ---- Module `iterators_checker' is ON
  -- The PAPI Library is found at PAPI_LIBRARY-NOTFOUND
  -- Module papi not selectable: PARSEC_PROF_TRACE disabled.
  -- ---- Module `print_steals' is ON
  -- ---- Module `ptg_to_dtd' is ON
  -- Module task_profiler not selectable: PARSEC_PROF_TRACE disabled.
  -- Component pins sources: mca/pins/pins.c;mca/pins/pins_init.c
  -- -- Found Component `sched'
  -- ---- Module `ap' is ON
  -- ---- Module `gd' is ON
  -- ---- Module `ip' is ON
  -- ---- Module `lfq' is ON
  -- ---- Module `lhq' is ON
  -- ---- Module `ll' is ON
  -- ---- Module `ltq' is ON
  -- ---- Module `pbq' is ON
  -- ---- Module `rnd' is ON
  -- ---- Module `spq' is ON
  -- Component sched sources:
  -- PARSEC Modular Component Architecture (MCA) discovery done.
  -- Could NOT find Omega; Options depending on Omega will be disabled (missing: OMEGA_INCLUDE_DIR OMEGA_LIBRARY)
  -- Detecting Fortran/C Interface
  -- Detecting Fortran/C Interface - Found GLOBAL and MODULE mangling
  -- Looking for PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
  -- Looking for PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128 - found
  -- Internal PaRSEC uses CAS 128B. Reconfiguring parsec_options.h


  Configuration flags:
    CMAKE_C_FLAGS          =  -m64 -std=c1x
    CMAKE_C_LDFLAGS        =  -m64
    CMAKE_EXE_LINKER_FLAGS =
    EXTRA_LIBS             = -latomic;/opt/local/lib/libhwloc.dylib;-L/opt/local/lib/gcc7/gcc/x86_64-apple-darwin18/7.4.0;-L/opt/local/lib/gcc7;-L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/usr/lib;gfortran;gcc_ext.10.5;gcc;quadmath;m



  -- Configuring done
  -- Generating done
  -- Build files have been written to:

If this is done, congratulations, PaRSEC is configured and you're ready for
building and testing the system.

Troubleshooting
---------------

In the unlikely case something goes wrong, read carefully the error message. We
spend a significant amount of time trying to output something meaningful for you
and for us (in case you need help to debug/understand). If the output is not
helpful enough to fix the problem, you should contact us via the PaRSEC user
mailing list and provide the CMake command and the flags, the output as well as
the files CMakeFiles/CMakeError.log and CMakeFiles/CMakeOutput.log.

We use quite a few packages that are optional, don't panic if they are not found
during the configuration. However, some of them are critical for increasing the
performance (such as HWLOC).

Check that you have a working MPI somewhere accessible (``mpicc`` and ``mpirun`` should
be in your PATH, except on Cray systems where you should use the ``cc`` wrapper).

If you have strange behavior, check that you have a success line for one of the
possible atomic backends that make sense for your local environment (i.e.,
C11 or GNU atomics depending on GCC versions, XLC on BlueGene machines, etc.).
If not, the atomic operations will not work, and that is damageable for the good
operation of PaRSEC. Note how in the shown configuration below, it takes
several attempts to get the right flags to use 128 bits atomic operations, but
in the end all looks good here.

.. code:: console

  -- Found target X86_64
  ...
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Failed
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Failed
  -- Performing Test PARSEC_ATOMIC_USE_C11_128
  -- Performing Test PARSEC_ATOMIC_USE_C11_128 - Success
  --       support for 32 bits atomics - found
  --       support for 64 bits atomics - found
  --       support for 128 bits atomics - found

CMake behavior can be modified from what your environment variables contain.
For example environment modules_, a popular way to load software on Cray,
DOE and NERSC supercomputers, can set many variables that will change the
outcome of the CMake configuration stage.

CC
  to choose your C compiler
CFLAGS
  to change your C compilation flags
LDFLAGS
  to change your C linking flags
FC
  to choose your Fortran compiler
XXX_DIR
  CMake FindXXX will try this directory as a priority
XXX_ROOT
  CMake FindXXX will include this directory in the search

.. _modules: https://www.nersc.gov/users/software/user-environment/modules/

Tuning the configuration: ccmake
--------------------------------

When the configuration is successful, you can tune it using ccmake:

.. code: shell
  ccmake .

(notice the double c of ``ccmake``). This is an interactive tool, that lets you
choose the compilation parameters. Navigate with the arrows to the parameter you
want to change and hit enter to edit. Remember that any changes will be lost
when you invoke again a ``configure`` script.

Notable parameters are::

  PARSEC_DEBUG                    OFF (and all other PARSEC_DEBUG options)
  PARSEC_DIST_COLLECTIVES         ON
  PARSEC_DIST_WITH_MPI            ON
  PARSEC_GPU_WITH_CUDA            ON
  PARSEC_OMEGA_DIR                OFF
  PARSEC_PROF_*                   OFF (all PARSEC_PROF_ flags off)

Using the *expert* mode (key 't' to toggle to expert mode), you can change other
useful options, like::

  CMAKE_C_FLAGS_RELEASE
  CMAKE_EXE_LINKER_FLAGS_RELEASE
  CMAKE_Fortran_FLAGS_RELEASE
  CMAKE_VERBOSE_MAKEFILE

And others to change the path to some compilers, for example. The
``CMAKE_VERBOSE_MAKEFILE`` option, when turned ``ON``, will display the command run when
compiling, which can help debugging configuration mistakes.  When you have set
all the options you want in ccmake, type 'c' to configure again, and 'g' to
generate the files. If you entered wrong values in some fields, ccmake will
complain at 'c' time.

Building PaRSEC
===============

If the configuration was good, compilation should be as simple and
fancy as ``make``. To debug issues, use ``make VERBOSE=1`` or turn the
``CMAKE_VERBOSE_MAKEFILE`` option to ``ON`` using ``ccmake``. Check
your compilation lines, and adapt your configuration options accordingly.

Spack
-----

Some DOE sites are exploring the use of Spack_ to install software. You
can integrate PaRSEC in a Spack environment by using the provided
configurations in ``contrib/spack``. See the Readme there for more details.

Running with PaRSEC
===================

.. code:: bash

  mpiexec -n 8 ./some_parsec_app

TL;DR
=====

.. code:: bash

  mkdir builddir && cd builddir
  ${srcdir}/configure --with-hwloc --with-mpi --disable-debug --prefix=$PWD/install
  make install
  mpiexec -n 8 examples/ex00

______

--
Happy hacking,
  The PaRSEC team.

