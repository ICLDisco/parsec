PROGRAM TOUCH_EXF

  use, INTRINSIC :: ISO_C_BINDING, only : c_int
  use dague_f08_interfaces

interface touch_initialize
  function touch_initialize_f08(block, n) BIND(C, name="touch_initialize")
    use, INTRINSIC :: ISO_C_BINDING, only : c_int
    use dague_f08_interfaces
    implicit none
    integer(kind=c_int), INTENT(IN), VALUE :: block
    integer(kind=c_int), INTENT(IN), VALUE :: n
    type(dague_handle_t)  :: touch_initialize_f08
  end function
end interface

interface touch_finalize
  function touch_finalize_f08() BIND(C, name="touch_finalize")
    use, INTRINSIC :: ISO_C_BINDING, only : c_int
    integer(kind=c_int) :: touch_finalize
  end function
end interface

  integer BLOCK, N, ret
  parameter (BLOCK=10, N=100)

  type(dague_context_t) :: context
  type(dague_handle_t)  :: handle

  call dague_init(1, context)

  handle = touch_initialize_f08(BLOCK, N)

  call dague_enqueue( context, handle )

  call dague_context_wait(context)

  call dague_fini(context)

  ret = touch_finalize_f08()

  call exit(ret)
END
