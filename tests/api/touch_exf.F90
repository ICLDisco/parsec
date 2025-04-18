!
! Copyright (c) 2021-2024 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
!

PROGRAM TOUCH_EXF

  use, INTRINSIC :: ISO_C_BINDING, only : c_int
  use parsec_f08_interfaces
  use mpi

interface
  function touch_initialize_f08(block, n) BIND(C, name="touch_initialize")
    use, INTRINSIC :: ISO_C_BINDING, only : c_int
    use parsec_f08_interfaces
    implicit none
    integer(kind=c_int), INTENT(IN), VALUE :: block
    integer(kind=c_int), INTENT(IN), VALUE :: n
    type(parsec_taskpool_t)  :: touch_initialize_f08
  end function touch_initialize_f08

  function touch_finalize_f08() BIND(C, name="touch_finalize")
    use, INTRINSIC :: ISO_C_BINDING, only : c_int
    integer(kind=c_int) :: touch_finalize_f08
  end function touch_finalize_f08
end interface

  integer BLOCK, N, mpith, ret
  parameter (BLOCK=10, N=100)

  type(parsec_context_t) :: context
  type(parsec_taskpool_t)  :: tp

  call MPI_Init_thread(MPI_THREAD_MULTIPLE, mpith, ret)

  call parsec_init(1, context)

  tp = touch_initialize_f08(BLOCK, N)

  call parsec_context_start(context, ret)

  call parsec_context_add_taskpool( context, tp, ret )

  call parsec_context_wait(context, ret)

  call parsec_taskpool_free(tp)

  call parsec_fini(context)

  ret = touch_finalize_f08()

  call MPI_Finalize(ret)

  call exit(ret)
END
