! -*- f90 -*-
! Copyright (c) 2013-2014 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module parsec_f08_interfaces

    use, intrinsic :: ISO_C_BINDING

    type, BIND(C) :: parsec_taskpool_t
      TYPE(C_PTR) :: PTR
    end type parsec_taskpool_t

    type, BIND(C) :: parsec_context_t
      TYPE(C_PTR) :: PTR
    end type parsec_context_t

ABSTRACT INTERFACE
SUBROUTINE parsec_event_cb(tp, cbdata) BIND(C)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), INTENT(IN) :: tp
    TYPE(C_PTR), INTENT(IN)          :: cbdata
END SUBROUTINE
END INTERFACE

INTERFACE parsec_init_f08
SUBROUTINE parsec_init_f08(nbcores, ctx, ierr) &
         BIND(C, name="parsec_init_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_context_t
    IMPLICIT NONE
    INTEGER(KIND=c_int), VALUE, INTENT(IN)  :: nbcores
    TYPE(parsec_context_t), INTENT(OUT)      :: ctx
    INTEGER(KIND=c_int), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_init_f08
END INTERFACE parsec_init_f08

INTERFACE parsec_fini_f08
SUBROUTINE parsec_fini_f08(context,ierr) &
           BIND(C, name="parsec_fini_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_context_t
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(INOUT)    :: context
    INTEGER(KIND=c_int), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_fini_f08
END INTERFACE parsec_fini_f08

INTERFACE parsec_compose_f08
FUNCTION parsec_compose_f08(start, next) &
         BIND(C, name="parsec_compose")
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: start
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: next
    TYPE(parsec_taskpool_t)                    :: parsec_compose_f08
END FUNCTION parsec_compose_f08
END INTERFACE parsec_compose_f08

INTERFACE parsec_taskpool_free_f08
SUBROUTINE parsec_taskpool_free_f08(ctx) &
         BIND(C, name="parsec_taskpool_free")
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: ctx
END SUBROUTINE parsec_taskpool_free_f08
END INTERFACE parsec_taskpool_free_f08

INTERFACE parsec_enqueue_f08
FUNCTION parsec_enqueue_f08(context, tp) &
           BIND(C, name="parsec_enqueue")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_taskpool_t, parsec_context_t
    IMPLICIT NONE
    TYPE(parsec_context_t), VALUE, INTENT(IN)  :: context
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    INTEGER(KIND=c_int)                        :: parsec_enqueue_f08
END FUNCTION parsec_enqueue_f08
END INTERFACE parsec_enqueue_f08

INTERFACE parsec_context_wait_f08
FUNCTION parsec_context_wait_f08(context) &
           BIND(C, name="parsec_context_wait")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_context_t
    IMPLICIT NONE
    TYPE(parsec_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: parsec_context_wait_f08
END FUNCTION parsec_context_wait_f08
END INTERFACE parsec_context_wait_f08

INTERFACE parsec_context_start_f08
FUNCTION parsec_context_start_f08(context) &
           BIND(C, name="parsec_context_start")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_context_t
    IMPLICIT NONE
    TYPE(parsec_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: parsec_context_start_f08
END FUNCTION parsec_context_start_f08
END INTERFACE parsec_context_start_f08

INTERFACE parsec_context_test_f08
FUNCTION parsec_context_test_f08(context) &
           BIND(C, name="parsec_context_test")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_context_t
    IMPLICIT NONE
    TYPE(parsec_context_t), VALUE, INTENT(IN)    :: context
    INTEGER(KIND=c_int)                         :: parsec_context_test_f08
END FUNCTION parsec_context_test_f08
END INTERFACE parsec_context_test_f08

INTERFACE  parsec_set_complete_callback_f08
SUBROUTINE parsec_set_complete_callback_f08(tp, complete_cb, &
                                            complete_data, ierr) &
           BIND( C, name="parsec_set_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    TYPE(C_FUNPTR), INTENT(IN)              :: complete_cb
    TYPE(C_PTR), INTENT(IN)                 :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_set_complete_callback_f08
END INTERFACE  parsec_set_complete_callback_f08

INTERFACE  parsec_get_complete_callback_f08
SUBROUTINE parsec_get_complete_callback_f08(tp, complete_cb, &
                                           complete_data, ierr) &
           BIND(C, name="parsec_get_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    TYPE(C_FUNPTR), INTENT(OUT)             :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_get_complete_callback_f08
END INTERFACE  parsec_get_complete_callback_f08

INTERFACE  parsec_set_enqueue_callback_f08
SUBROUTINE parsec_set_enqueue_callback_f08(tp, enqueue_cb, &
                                          enqueue_data, ierr) &
           BIND( C, name="parsec_set_enqueue_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    TYPE(C_FUNPTR), INTENT(IN)              :: enqueue_cb
    TYPE(C_PTR), INTENT(IN)                 :: enqueue_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_set_enqueue_callback_f08
END INTERFACE  parsec_set_enqueue_callback_f08

INTERFACE  parsec_get_enqueue_callback_f08
SUBROUTINE parsec_get_enqueue_callback_f08(tp, enqueue_cb, &
                                          enqueue_data, ierr) &
           BIND(C, name="parsec_get_enqueue_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    TYPE(C_FUNPTR), INTENT(OUT)             :: enqueue_cb
    TYPE(C_PTR), INTENT(OUT)                :: enqueue_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_get_enqueue_callback_f08
END INTERFACE  parsec_get_enqueue_callback_f08

INTERFACE  parsec_set_priority_f08
SUBROUTINE parsec_set_priority_f08(tp, priority, &
           ierr) BIND( C, name="parsec_set_priority_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT parsec_taskpool_t
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), VALUE, INTENT(IN) :: tp
    INTEGER(KIND=C_INT), VALUE, INTENT(IN)  :: priority
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE parsec_set_priority_f08
END INTERFACE  parsec_set_priority_f08

CONTAINS

SUBROUTINE parsec_init(nbcores, ctx, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER(KIND=c_int), VALUE, INTENT(IN)     :: nbcores
    TYPE(parsec_context_t), INTENT(OUT)         :: ctx
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call parsec_init_f08(nbcores, ctx, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_init

SUBROUTINE parsec_fini(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(INOUT)       :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call parsec_fini_f08(context, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_fini

SUBROUTINE parsec_taskpool_free(ctx, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_taskpool_t), INTENT(IN) :: ctx
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr

    call parsec_taskpool_free_f08(ctx)
    if(present(ierr)) ierr = 0
END SUBROUTINE parsec_taskpool_free

SUBROUTINE parsec_enqueue(context, tp, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(IN)          :: context
    TYPE(parsec_taskpool_t), INTENT(IN)         :: tp
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
    INTEGER(KIND=C_INT)                         :: c_err

    c_err = parsec_enqueue_f08(context, tp)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_enqueue

SUBROUTINE parsec_context_wait(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    c_err = parsec_context_wait_f08(context)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_context_wait

SUBROUTINE parsec_context_start(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    c_err = parsec_context_start_f08(context)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_context_start

SUBROUTINE parsec_context_test(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), INTENT(OUT)           :: ierr

    ierr = parsec_context_test_f08(context)
END SUBROUTINE parsec_context_test

SUBROUTINE parsec_set_complete_callback(tp, complete_cb, &
                                        complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPLICIT NONE
    TYPE(parsec_taskpool_t)                    :: tp
    PROCEDURE(parsec_event_cb), BIND(C)        :: complete_cb
    TYPE(C_PTR), INTENT(IN)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    TYPE(C_FUNPTR)                             :: c_fct
    INTEGER(KIND=C_INT)                        :: c_err

    c_fct = C_FUNLOC(complete_cb)
    call parsec_set_complete_callback_f08(tp, c_fct, &
                                          complete_data, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_set_complete_callback

SUBROUTINE parsec_get_complete_callback(tp, complete_cb, &
                                        complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(parsec_taskpool_t)                     :: tp
    PROCEDURE(parsec_event_cb), POINTER, INTENT(OUT) :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
    TYPE(C_FUNPTR)                              :: c_fun
    INTEGER(KIND=C_INT)                         :: c_err

    call parsec_get_complete_callback_f08(tp, c_fun, &
                                          complete_data, c_err)
    call C_F_PROCPOINTER(c_fun, complete_cb)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_get_complete_callback

SUBROUTINE parsec_set_enqueue_callback(tp, enqueue_cb, &
                                       enqueue_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_FUNPTR
    IMPLICIT NONE
    TYPE(parsec_taskpool_t)                    :: tp
    PROCEDURE(parsec_event_cb), BIND(C)        :: enqueue_cb
    TYPE(C_PTR), INTENT(IN)                    :: enqueue_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    TYPE(C_FUNPTR)                             :: c_fct
    INTEGER(KIND=C_INT)                        :: c_err

    c_fct = C_FUNLOC(enqueue_cb)
    call parsec_set_enqueue_callback_f08(tp, c_fct, &
                                         enqueue_data, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_set_enqueue_callback

SUBROUTINE parsec_get_enqueue_callback(tp, enqueue_cb, &
                                       enqueue_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(parsec_taskpool_t)                     :: tp
    PROCEDURE(parsec_event_cb), POINTER, INTENT(OUT) :: enqueue_cb
    TYPE(C_PTR), INTENT(OUT)                    :: enqueue_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
    TYPE(C_FUNPTR)                              :: c_fun
    INTEGER(KIND=C_INT)                         :: c_err

    call parsec_get_enqueue_callback_f08(tp, c_fun, &
                                         enqueue_data, c_err)
    call C_F_PROCPOINTER(c_fun, enqueue_cb)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_get_enqueue_callback

SUBROUTINE parsec_set_priority(tp, priority, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(parsec_taskpool_t)                    :: tp
    INTEGER(KIND=C_INT), VALUE, INTENT(IN)     :: priority
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call parsec_set_priority_f08(tp, priority, c_err)
    if(present(ierr)) ierr = c_err
END SUBROUTINE parsec_set_priority

end module parsec_f08_interfaces

