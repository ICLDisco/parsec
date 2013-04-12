! -*- f90 -*-
! Copyright (c) 2013      The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module dague_f08

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

INTERFACE dague_init
FUNCTION dague_init_f08(nbcores, argc, argv) &
         BIND(C, name="dague_init")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT, C_SIGNED_CHAR
    IMPORT dague_context_t
    IMPLICIT NONE
    INTEGER(KIND=c_int), VALUE, INTENT(IN)    :: nbcores
    INTEGER(KIND=c_int), INTENT(INOUT)        :: argc
    CHARACTER(KIND=C_SIGNED_CHAR),INTENT(IN)  :: argv(argc,*)
    TYPE(dague_context_t)                     :: dague_init_f08
END FUNCTION dague_init_f08
END INTERFACE dague_init

INTERFACE dague_fini
SUBROUTINE dague_fini_f08(context,ierr) &
           BIND(C, name="dague_fini")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(INOUT) :: context
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT)       :: ierr
END SUBROUTINE dague_fini_f08
END INTERFACE dague_fini

INTERFACE dague_compose
FUNCTION dague_compose_f08(start, next) &
         BIND(C, name="dague_compose")
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t), INTENT(IN) :: start
    TYPE(dague_object_t), INTENT(IN) :: next
    TYPE(dague_object_t)             :: dague_compose_f08
END FUNCTION dague_compose_f08
END INTERFACE dague_compose

INTERFACE dague_enqueue
SUBROUTINE dague_enqueue_f08(context, object, ierr) &
           BIND(C, name="dague_enqueue")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t, dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)           :: context
    TYPE(dague_object_t), INTENT(IN)            :: object
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT)  :: ierr
END SUBROUTINE dague_enqueue_f08
END INTERFACE dague_enqueue

INTERFACE dague_progress
SUBROUTINE dague_progress_f08(context, ierr) &
           BIND(C, name="dague_progress")
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPORT dague_context_t
    IMPLICIT NONE
    TYPE(dague_context_t), INTENT(IN)          :: context
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
END SUBROUTINE dague_progress_f08
END INTERFACE dague_progress

INTERFACE  dague_set_complete_callback
SUBROUTINE dague_set_complete_callback_f08(object, complete_cb, &
                                           complete_data, ierr) &
           BIND( C, name="dague_set_complete_callback")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t, dague_completion_cb
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    PROCEDURE(dague_completion_cb)             :: complete_cb
    TYPE(C_PTR), INTENT(IN)                    :: complete_data
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
END SUBROUTINE dague_set_complete_callback_f08
END INTERFACE  dague_set_complete_callback

INTERFACE  dague_get_complete_callback
SUBROUTINE dague_get_complete_callback_f08(object, complete_cb, &
                                           complete_data, ierr) &
           BIND(C, name="dague_get_complete_callback")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t, dague_completion_cb
    IMPLICIT NONE
    TYPE(dague_object_t)                        :: object
    TYPE(C_PTR), INTENT(OUT)                    :: complete_cb
    TYPE(C_PTR), INTENT(OUT)                    :: complete_data
    INTEGER(KIND=C_INT), OPTIONAL, INTENT(OUT)  :: ierr
END SUBROUTINE dague_get_complete_callback_f08
END INTERFACE  dague_get_complete_callback

INTERFACE  dague_object_start
SUBROUTINE dague_object_start_f08(object, &
           ierr) BIND( C, name="dague_object_start")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
END SUBROUTINE dague_object_start_f08
END INTERFACE  dague_object_start

INTERFACE  dague_set_priority
SUBROUTINE dague_set_priority_f08(object, priority, &
           ierr) BIND( C, name="dague_set_priority")
    USE, intrinsic :: ISO_C_BINDING, only : C_PTR, C_INT
    IMPORT dague_object_t
    IMPLICIT NONE
    TYPE(dague_object_t)                       :: object
    INTEGER(KIND=c_int), VALUE, INTENT(IN)     :: priority
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
END SUBROUTINE dague_set_priority_f08
END INTERFACE  dague_set_priority

end module dague_f08

