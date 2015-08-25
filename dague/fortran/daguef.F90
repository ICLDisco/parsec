! -*- f90 -*-
! Copyright (c) 2013-2014 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module dague_f08_interfaces

    use, intrinsic :: ISO_C_BINDING

    type, BIND(C) :: dague_handle_t
      TYPE(C_PTR) :: PTR
    end type dague_handle_t

    type, BIND(C) :: dague_context_t
      TYPE(C_PTR) :: PTR
    end type dague_context_t

ABSTRACT INTERFACE
SUBROUTINE dague_event_cb(handle, cbdata) BIND(C)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), INTENT(IN) :: handle
    TYPE(C_PTR), INTENT(IN)          :: cbdata
END SUBROUTINE
END INTERFACE

INTERFACE dague_init_f08
SUBROUTINE dague_init_f08(nbcores, ctx, ierr) &
         BIND(C, name="dague_init_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    INTEGER(KIND=c_int), VALUE, INTENT(IN)  :: nbcores
    TYPE(dague_context_t), INTENT(OUT)      :: ctx
    INTEGER(KIND=c_int), INTENT(OUT)        :: ierr
END SUBROUTINE dague_init_f08
END INTERFACE dague_init_f08

INTERFACE dague_fini_f08
SUBROUTINE dague_fini_f08(context,ierr) &
           BIND(C, name="dague_fini_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(INOUT)    :: context
    INTEGER(KIND=c_int), INTENT(OUT)        :: ierr
END SUBROUTINE dague_fini_f08
END INTERFACE dague_fini_f08

INTERFACE dague_compose_f08
FUNCTION dague_compose_f08(start, next) &
         BIND(C, name="dague_compose")
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: start
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: next
    TYPE(dague_handle_t)                    :: dague_compose_f08
END FUNCTION dague_compose_f08
END INTERFACE dague_compose_f08

INTERFACE dague_handle_free_f08
SUBROUTINE dague_handle_free_f08(ctx) &
         BIND(C, name="dague_handle_free")
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: ctx
END SUBROUTINE dague_handle_free_f08
END INTERFACE dague_handle_free_f08

INTERFACE dague_enqueue_f08
FUNCTION dague_enqueue_f08(context, handle) &
           BIND(C, name="dague_enqueue")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_handle_t, dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), VALUE, INTENT(IN)  :: context
    TYPE(dague_handle_t), VALUE, INTENT(IN)   :: handle
    INTEGER(KIND=c_int)                       :: dague_enqueue_f08
END FUNCTION dague_enqueue_f08
END INTERFACE dague_enqueue_f08

INTERFACE dague_context_wait_f08
FUNCTION dague_context_wait_f08(context) &
           BIND(C, name="dague_context_wait")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: dague_context_wait_f08
END FUNCTION dague_context_wait_f08
END INTERFACE dague_context_wait_f08

INTERFACE dague_context_start_f08
FUNCTION dague_context_start_f08(context) &
           BIND(C, name="dague_context_start")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: dague_context_start_f08
END FUNCTION dague_context_start_f08
END INTERFACE dague_context_start_f08

INTERFACE dague_context_test_f08
FUNCTION dague_context_test_f08(context) &
           BIND(C, name="dague_context_test")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: dague_context_test_f08
END FUNCTION dague_context_test_f08
END INTERFACE dague_context_test_f08

INTERFACE  dague_set_complete_callback_f08
SUBROUTINE dague_set_complete_callback_f08(handle, complete_cb, &
                                           complete_data, ierr) &
           BIND( C, name="dague_set_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: handle
    TYPE(C_FUNPTR), INTENT(IN)              :: complete_cb
    TYPE(C_PTR), INTENT(IN)                 :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_set_complete_callback_f08
END INTERFACE  dague_set_complete_callback_f08

INTERFACE  dague_get_complete_callback_f08
SUBROUTINE dague_get_complete_callback_f08(handle, complete_cb, &
                                           complete_data, ierr) &
           BIND(C, name="dague_get_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: handle
    TYPE(C_FUNPTR), INTENT(OUT)             :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_get_complete_callback_f08
END INTERFACE  dague_get_complete_callback_f08

INTERFACE  dague_set_enqueue_callback_f08
SUBROUTINE dague_set_enqueue_callback_f08(handle, enqueue_cb, &
                                          enqueue_data, ierr) &
           BIND( C, name="dague_set_enqueue_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: handle
    TYPE(C_FUNPTR), INTENT(IN)              :: enqueue_cb
    TYPE(C_PTR), INTENT(IN)                 :: enqueue_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_set_enqueue_callback_f08
END INTERFACE  dague_set_enqueue_callback_f08

INTERFACE  dague_get_enqueue_callback_f08
SUBROUTINE dague_get_enqueue_callback_f08(handle, enqueue_cb, &
                                          enqueue_data, ierr) &
           BIND(C, name="dague_get_enqueue_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: handle
    TYPE(C_FUNPTR), INTENT(OUT)             :: enqueue_cb
    TYPE(C_PTR), INTENT(OUT)                :: enqueue_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_get_enqueue_callback_f08
END INTERFACE  dague_get_enqueue_callback_f08

INTERFACE  dague_set_priority_f08
SUBROUTINE dague_set_priority_f08(handle, priority, &
           ierr) BIND( C, name="dague_set_priority_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_handle_t
    IMPLICIT NONE
    TYPE(dague_handle_t), VALUE, INTENT(IN) :: handle
    INTEGER(KIND=C_INT), VALUE, INTENT(IN)  :: priority
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_set_priority_f08
END INTERFACE  dague_set_priority_f08

CONTAINS

SUBROUTINE dague_init(nbcores, ctx, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER(KIND=c_int), VALUE, INTENT(IN)     :: nbcores
    TYPE(dague_context_t), INTENT(OUT)         :: ctx
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_init_f08(nbcores, ctx, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_init

SUBROUTINE dague_fini(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(INOUT)       :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_fini_f08(context, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_fini

SUBROUTINE dague_handle_free(ctx, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_handle_t), INTENT(IN) :: ctx
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr

    call dague_handle_free_f08(ctx)
    if(present(ierr)) ierr = 0
END SUBROUTINE dague_handle_free

SUBROUTINE dague_enqueue(context, handle, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    TYPE(dague_handle_t), INTENT(IN)           :: handle
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    c_err = dague_enqueue_f08(context, handle)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_enqueue

SUBROUTINE dague_context_wait(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    c_err = dague_context_wait_f08(context)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_context_wait

SUBROUTINE dague_context_start(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    c_err = dague_context_start_f08(context)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_context_start

SUBROUTINE dague_context_test(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), INTENT(OUT)           :: ierr

    ierr = dague_context_test_f08(context)
END SUBROUTINE dague_context_test

SUBROUTINE dague_set_complete_callback(handle, complete_cb, &
                                       complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPLICIT NONE
    TYPE(dague_handle_t)                       :: handle
    PROCEDURE(dague_event_cb), BIND(C)         :: complete_cb
    TYPE(C_PTR), INTENT(IN)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    TYPE(C_FUNPTR)                             :: c_fct
    INTEGER(KIND=C_INT)                        :: c_err

    c_fct = C_FUNLOC(complete_cb)
    call dague_set_complete_callback_f08(handle, c_fct, &
                                         complete_data, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_set_complete_callback

SUBROUTINE dague_get_complete_callback(handle, complete_cb, &
                                       complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(dague_handle_t)                        :: handle
    PROCEDURE(dague_event_cb), POINTER, INTENT(OUT) :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
    TYPE(C_FUNPTR)                              :: c_fun
    INTEGER(KIND=C_INT)                         :: c_err

    call dague_get_complete_callback_f08(handle, c_fun, &
                                         complete_data, c_err)
    call C_F_PROCPOINTER(c_fun, complete_cb)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_get_complete_callback

SUBROUTINE dague_set_enqueue_callback(handle, enqueue_cb, &
                                      enqueue_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPLICIT NONE
    TYPE(dague_handle_t)                       :: handle
    PROCEDURE(dague_event_cb), BIND(C)         :: enqueue_cb
    TYPE(C_PTR), INTENT(IN)                    :: enqueue_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    TYPE(C_FUNPTR)                             :: c_fct
    INTEGER(KIND=C_INT)                        :: c_err

    c_fct = C_FUNLOC(enqueue_cb)
    call dague_set_enqueue_callback_f08(handle, c_fct, &
                                        enqueue_data, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_set_enqueue_callback

SUBROUTINE dague_get_enqueue_callback(handle, enqueue_cb, &
                                      enqueue_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(dague_handle_t)                        :: handle
    PROCEDURE(dague_event_cb), POINTER, INTENT(OUT) :: enqueue_cb
    TYPE(C_PTR), INTENT(OUT)                    :: enqueue_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
    TYPE(C_FUNPTR)                              :: c_fun
    INTEGER(KIND=C_INT)                         :: c_err

    call dague_get_enqueue_callback_f08(handle, c_fun, &
                                        enqueue_data, c_err)
    call C_F_PROCPOINTER(c_fun, enqueue_cb)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_get_enqueue_callback

SUBROUTINE dague_set_priority(handle, priority, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_handle_t)                       :: handle
    INTEGER(KIND=C_INT), VALUE, INTENT(IN)     :: priority
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_set_priority_f08(handle, priority, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE dague_set_priority

end module dague_f08_interfaces

