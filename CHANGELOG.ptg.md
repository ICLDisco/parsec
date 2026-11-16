Changelog for features specific to the PTG Domain Specific Language


2026-11-16

 - The inline C code of a dependency property (type, layout, count, displ and
   their _remote and _data variants) receives a description of the dependency,
   dep_ctx (parsec_dep_context_t, parsec/runtime.h): the task on the other side
   (parent) and the size of the incoming data (incoming_bytes). A receive can
   be sized after what the producer sent, into the application's own arena:
         B <- B SEND(k) [type_remote = DBL layout_remote = parsec_datatype_double_t
                         count_remote = %{ return dep_ctx->incoming_bytes / sizeof(double); %}]
 - A receive is allocated as the number of elements of its arena that hold it.
   Release note: when the successors of one payload disagree on the datatype
   (the packed fallback), the receive takes the packed bytes rounded up to
   elements of the arena, no longer the packed bytes times the element size.
   When they agree on the datatype but not on the count, the largest count is
   received, whatever their order (the last one used to win).

2020-12-22

 - Add PTG CUDA bodies support for:
      - User-defined stage_in/stage_out routines for data transfers.
      - User-defined size of buffers allocated on the GPU memory.
      - Data collection for which the flow logically belongs to, enables
         user-defined stage in/out routines to rely on the data collection
         to get info necessary for the transfer.
      BODY [type=CUDA
         stage_in=< user routine >
         stage_in=< user routine >
         < flowname >.size=< bytes to be reserve on GPU >
         < flowname >.dc=< datacollection for this flow >]

 - PaRSEC data copies are described by a type.
   Set up by data collection when accessing data or when creating temporary
   copies.

 - Change PTG typing of dependencies to specify datatype used to send/receive
   data between remote peers using the keyword [type_remote =...]
         TASK_A: A -> B TASK_B [type_remote = t1]
         TASK_B: B <- A TASK_A [type_remote = t2]
   * Data is sent with type_remote on the output dependency of the sender and
     received on the receiver using the type_remote on the input dependency.
   * If no types are specified, data is sent using the data copy type and
     received using the default type of the taskpool arena.

 - PTG supports reshaping of datacopies (type conversion):
   * During accesses to data collections:
      - Reading from matrix: A <- desc(m, n) [type =... type_data=...]
         If !undef(type) && !undef(type_data): Pack desc(m,n) type_data, Unpack type
         If undef(type)  && !undef(type_data): Pack desc(m,n) type_data, Unpack type_data
         If !undef(type) && undef(type_data):  Pack desc(m,n) desc(m,n).type, Unpack type
         If undef(type)  && undef(type_data):  no action
      - Writing to matrix:   A -> desc(m, n) [type =.. type_data=…]
         If !undef(type) && !undef(type_data): Pack A type,   Unpack type_data on desc(m,n)
         If undef(type)  && !undef(type_data): Pack A A.type, Unpack type_data on desc(m,n)
         If !undef(type) && undef(type_data):  Pack A type,   Unpack desc(m,n).type on desc(m,n)
         If undef(type)  && undef(type_data):  Pack A A.type, Unpack desc(m,n).type on desc(m,n)
   * During propagation of dependencies between local tasks using
     keyword [type =...] on output and input dependencies.
      TASK_A: A -> B TASK_B
      TASK_B: B <- A TASK_A
          No type, no reshape
      ---
      TASK_A: A -> B TASK_B
      TASK_B: B <- A TASK_A [type = t2]
          Pack A.dtt, Unpack t2
      ---
      TASK_A: A -> B TASK_B [type = t1]
      TASK_B: B <- A TASK_A [type = t2]
          Pack t1, Unpack t2
      ---
      TASK_A: A -> B TASK_B [type = t1]
      TASK_B: B <- A TASK_A
          Pack t1, Unpack t1

  - NEW type specify by [type = ... ]


