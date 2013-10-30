! -*- f90 -*-
! Copyright (c) 2013      The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module dague_f08_interfaces

    use, intrinsic :: ISO_C_BINDING

    type, BIND(C) :: dague_object_t
      TYPE(C_PTR) :: PTR
    end type dague_object_t

    type, BIND(C) :: dague_context_t
      TYPE(C_PTR) :: PTR
    end type dague_context_t

ABSTRACT INTERFACE
SUBROUTINE dague_completion_cb(object, cbdata) BIND(C)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t), INTENT(IN) :: object
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
         BIND(C, name="dague_compose_f08")
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t), INTENT(IN) :: start
    TYPE(dague_object_t), INTENT(IN) :: next
    TYPE(dague_object_t)             :: dague_compose_f08
END FUNCTION dague_compose_f08
END INTERFACE dague_compose_f08

INTERFACE dague_enqueue_f08
SUBROUTINE dague_enqueue_f08(context, object, ierr) &
           BIND(C, name="dague_enqueue_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_object_t, dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)  :: context
    TYPE(dague_object_t), INTENT(IN)   :: object
    INTEGER(KIND=c_int), INTENT(OUT)   :: ierr
END SUBROUTINE dague_enqueue_f08
END INTERFACE dague_enqueue_f08

INTERFACE dague_progress_f08
SUBROUTINE dague_progress_f08(context, ierr) &
           BIND(C, name="dague_progress_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)    :: context
    INTEGER(KIND=c_int), INTENT(OUT)     :: ierr
END SUBROUTINE dague_progress_f08
END INTERFACE dague_progress_f08

INTERFACE  dague_set_complete_callback_f08
SUBROUTINE dague_set_complete_callback_f08(object, complete_cb, &
                                           complete_data, ierr) &
           BIND( C, name="dague_set_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t, dague_completion_cb
    IMPLICIT NONE
    TYPE(dague_object_t)                    :: object
    PROCEDURE(dague_completion_cb), POINTER :: complete_cb
    TYPE(C_PTR), INTENT(IN)                 :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_set_complete_callback_f08
END INTERFACE  dague_set_complete_callback_f08

INTERFACE  dague_get_complete_callback_f08
SUBROUTINE dague_get_complete_callback_f08(object, complete_cb, &
                                           complete_data, ierr) &
           BIND(C, name="dague_get_complete_callback_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t, dague_completion_cb
    IMPLICIT NONE
    TYPE(dague_object_t)                    :: object
    PROCEDURE(dague_completion_cb), POINTER :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                :: complete_data
    INTEGER(KIND=C_INT), INTENT(OUT)        :: ierr
END SUBROUTINE dague_get_complete_callback_f08
END INTERFACE  dague_get_complete_callback_f08

INTERFACE  dague_set_priority_f08
SUBROUTINE dague_set_priority_f08(object, priority, &
           ierr) BIND( C, name="dague_set_priority_f08")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t)                    :: object
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
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_init

SUBROUTINE dague_fini(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(INOUT)       :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_fini_f08(context, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_fini

SUBROUTINE dague_enqueue(context, object, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    TYPE(dague_object_t), INTENT(IN)           :: object
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_enqueue_f08(context, object, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_enqueue

SUBROUTINE dague_progress(context, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_progress_f08(context, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_progress

SUBROUTINE dague_set_complete_callback(object, complete_cb, &
                                       complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    PROCEDURE(dague_completion_cb), POINTER    :: complete_cb
    TYPE(C_PTR), INTENT(IN)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_set_complete_callback_f08(object, complete_cb, &
                                         complete_data, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_set_complete_callback

SUBROUTINE dague_get_complete_callback(object, complete_cb, &
                                       complete_data, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    PROCEDURE(dague_completion_cb), POINTER    :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                   :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_get_complete_callback_f08(object, complete_cb, &
                                         complete_data, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_get_complete_callback

SUBROUTINE dague_set_priority(object, priority, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    INTEGER(KIND=C_INT), VALUE, INTENT(IN)     :: priority
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=C_INT)                        :: c_err

    call dague_set_priority_f08(object, priority, c_err)
    if(present(ierr)) ierr = c_err;
END SUBROUTINE dague_set_priority

end module dague_f08_interfaces

