PaRSEC Documentation {#mainpage}
================================

# <a name="toc">Table of Contents</a>

A. [User Documentation](#parsecuser)
B. [Developer and Advanced Usage Documentation](#parsecdev)

## <a name="parsecuser">PaRSEC User Documentation</a> ###
[(Return to Table of Contents](#toc)]

This document introduces the programing interfaces of PaRSEC. For more
general information, see the
[PaRSEC bitbucket wiki page](https://bitbucket.org/icldistcomp/parsec/wiki/Home)
for information on how to compile and run PaRSEC. Also see the
[PaRSEC bitbucket wiki page](https://bitbucket.org/icldistcomp/parsec/wiki/Home)
for more general information on how to develop applications over
PaRSEC.

PaRSEC exposes a [public API](@ref parsec_public), that feature:

- a set of programming interfaces:

  - [Dynamic Tasks Discovery (DTD)](@ref DTD_INTERFACE), that uses an
  inspector/executor model to build the DAG of tasks at runtime
  - [Parameterized Task Graphs (PTG)](https://bitbucket.org/icldistcomp/parsec/wiki/writejdf), that
  provides an intermediate representation of the DAG of tasks at
  compile time
  
- [A runtime system](@ref parsec_public_runtime), to initialize the
task runtime system, schedule DAGs of Tasks in it, and expose the user
data to the task system
- [A profiling system](@ref parsec_public_profiling) to build traces
of the execution at runtime, providing performance information
feedback to the application or to the user

## <a name="parsecdev">PaRSEC Developer and Advanced Usage Documentation</a> ###
[(Return to Table of Contents](#toc)]

The code of the PaRSEC runtime is located in the `parsec/`
subdirectory of the source. It is separated in a few modules:

- Memory Management:

    - [Arenas](@ref parsec_internal_arena) represent temporary memory
    allocated by the runtime engine to move and store user data.
    - [Memory pools](@ref parsec_internal_mempool) are used to improve
   the performance of frequent allocation/de-allocation inside the
   runtime engine

- Meta Data Management:
   
   - [Data objects](@ref parsec_internal_data) represent the
   meta-information associated to each user's or temporary data blocks
   that the PaRSEC runtime engine manipulate.
   - [Data Repositories](@ref parsec_internal_datarepo) store
    [data objects](@ref parsec_internal_data) into hash tables for tasks
    to retrieve them when they become schedulable.

- Communication Management:

   - [the Communication Engine System](@ref parsec_internal_communication)
   holds all operations necessary to allow communicating control
   and data between processes in a distributed environment

- Computation Management:

    - [the Binding System](@ref parsec_internal_binding) allows to bind
   threads on multiple cores with a variety of interfaces for
   portability.
     - [Virtual Processes](@ref parsec_internal_virtualprocess) allow to
   isolate groups of threads and avoid work stealing between threads
   belonging to different virtual processes.
    - [The Internal Runtime Module](@ref parsec_internal_runtime) holds
    all other functions and data structures that allow to build the
    PaRSEC runtime system.

- Debugging and Tracing systems:

    - [The Debugging System](@ref parsec_internal_debug) holds functions
    and macros that are used internally to check internal assertions and
    output debugging information when requested by the user.
    - The [code documentation](@ref parsec_internal_profiling) for the
    [tracing system](@ref parsec_public_profiling) is also available.

- Basic Algorithms and data structures: the directory `parsec/class`
holds a set of [basic algorithms and data structures](@ref
parsec_internal_classes) that are used throughout the rest of the
code:

 - Base Classes:

    - [PaRSEC Objects](@ref parsec_internal_classes_object)
    - [List Items](@ref parsec_internal_classes_listitem)
	
 - Algorithms:
   
    - [Barrier](@ref parsec_internal_classes_barrier)
    - [Dequeue](@ref parsec_internal_classes_dequeue)
    - [FIFO](@ref parsec_internal_classes_fifo)
    - [Hash Tables](@ref parsec_internal_classes_hashtable)
    - [LIFO](@ref parsec_internal_classes_lifo)
    - [Linked Lists](@ref parsec_internal_classes_list)
    - [Value Arrays](@ref parsec_internal_classes_valuearray)

Some components of PaRSEC use the Modular Component Architecture (MCA)
initially developed for [Open MPI](https://www.open-mpi.org). This
enables a dynamic selection of components at runtime, based on the
hardware capability and the end-user choices. The
[MCA API](@ref parsec/mca/mca.h) enables the development of components and modules
inside frameworks. A component defines an API that is used by PaRSEC,
and a variety of modules implement this API.

Components defined with MCA are in the `parsec/mca` directory. The
following components have specific documentation:

 - [schedulers](@ref parsec/mca/sched/sched.h) in `parsec/mca/sched`
 - [PaRSEC INStrumentation](@ref parsec/mca/pins/pins.h) in `parsec/mca/pins`

