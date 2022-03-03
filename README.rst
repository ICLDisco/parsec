PaRSEC
======

PaRSEC: the Parallel Runtime Scheduler and Execution Controller for
micro-tasks on distributed heterogeneous systems.


Features
--------

PaRSEC_ is a generic framework for architecture aware scheduling and
management of micro-tasks on distributed many-core heterogeneous
architectures. Applications are expressed as a Direct Acyclic Graph of tasks
with labeled edges designating data dependencies. PaRSEC assigns computation
threads to the cores, overlaps communications and computations between nodes
as well as between host and accelerators (like GPUs). It achieves these
features by using a dynamic, fully-distributed scheduler based on architectural
features such as NUMA nodes and GPU awareness, as well as algorithmic features
such as data reuse.

Several high level languages are proposed to expose the DAG from the
applications. You can either build the DAG as you go, by using a mechanism
called dynamic task discovery (DTD), or use the Parameterized Task Graph
language (PTG) to expose a compact problem-size independent format that can
be queried on-demand to discover data dependencies in a totally distributed
fashion.

The framework includes libraries, a runtime system, and development tools to
help application developers tackle the difficult task of porting their
applications to highly heterogeneous and diverse environment.

.. _PaRSEC: https://github.com/icldisco/parsec


Installing PaRSEC
-----------------

Please read the INSTALL.rst_ file for the software dependencies and the
installation instructions.

.. _INSTALL.rst: https://github.com/icldisco/parsec/blob/master/INSTALL.rst


Links and Community
-------------------

You can report bugs on the GitHub issues_ and find documentation and
tutorials in the PaRSEC wiki_.

.. _issues: https://github.com/icldisco/parsec/issues
.. _wiki: https://github.com/icldisco/parsec/wiki

If you are interested in this project and want to join the users community,
our mailman will be happy to accept you on the project user (moderated)
mailing list at parsec-users@icl.utk.edu.


----

Happy hacking,
  The PaRSEC team.
