! -*- f90 -*-
! Copyright (c) 2013      The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module dague_profile_f08_interfaces

  use, intrinsic :: ISO_C_BINDING

  type, BIND(C) :: dague_thread_profiling_t
     TYPE(C_PTR) :: HANDLE
  end type dague_thread_profiling_t

  INTERFACE dague_profiling_init_f08
     SUBROUTINE dague_profiling_init_f08(hdr_id, ierr) &
          BIND(C, name="dague_profiling_init_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
       IMPLICIT NONE
       CHARACTER(KIND=c_char), INTENT(IN) :: hdr_id(*)
       INTEGER(KIND=c_int), INTENT(OUT)   :: ierr
     END SUBROUTINE dague_profiling_init_f08
  END INTERFACE dague_profiling_init_f08

  INTERFACE dague_profiling_fini_f08
     SUBROUTINE dague_profiling_fini_f08(ierr) &
          BIND(C, name="dague_profiling_fini_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=c_int), INTENT(OUT)   :: ierr
     END SUBROUTINE dague_profiling_fini_f08
  END INTERFACE dague_profiling_fini_f08

  INTERFACE dague_profiling_reset_f08
     SUBROUTINE dague_profiling_reset_f08(ierr) &
          BIND(C, name="dague_profiling_reset_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=c_int), INTENT(OUT)   :: ierr
     END SUBROUTINE dague_profiling_reset_f08
  END INTERFACE dague_profiling_reset_f08

  INTERFACE dague_profiling_dump_f08
     SUBROUTINE dague_profiling_dump_f08(fname, ierr) &
          BIND(C, name="dague_profiling_dump_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
       IMPLICIT NONE
       CHARACTER(KIND=c_char), INTENT(IN) :: fname(*)
       INTEGER(KIND=c_int), INTENT(OUT)   :: ierr
     END SUBROUTINE dague_profiling_dump_f08
  END INTERFACE dague_profiling_dump_f08

  INTERFACE dague_profiling_thread_init_f08
     FUNCTION dague_profiling_thread_init_f08(length, id_name) &
          BIND(C, name="dague_profiling_thread_init_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
       IMPORT dague_thread_profiling_t
       IMPLICIT NONE
       INTEGER(KIND=C_SIZE_T), VALUE, INTENT(IN) :: length
       CHARACTER(KIND=c_char), INTENT(IN)        :: id_name(*)
       TYPE(dague_thread_profiling_t)            :: dague_profiling_thread_init_f08
     END FUNCTION dague_profiling_thread_init_f08
  END INTERFACE dague_profiling_thread_init_f08

  INTERFACE dague_profiling_add_dictionary_keyword_08
     SUBROUTINE dague_profiling_add_dictionary_keyword_f08(key_name, &
          attr, info_length, conv_code, key_start, key_end, ierr) &
          BIND(C, name="dague_profiling_add_dictionary_keyword_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
       IMPLICIT NONE
       CHARACTER(KIND=c_char), INTENT(IN)        :: key_name(*)
       CHARACTER(KIND=c_char), INTENT(IN)        :: attr(*)
       INTEGER(KIND=C_SIZE_T), VALUE, INTENT(IN) :: info_length
       CHARACTER(KIND=c_char), INTENT(IN)        :: conv_code(*)
       INTEGER(KIND=c_int), INTENT(OUT)          :: key_start
       INTEGER(KIND=c_int), INTENT(OUT)          :: key_end
       INTEGER(KIND=c_int), INTENT(OUT)          :: ierr
     END SUBROUTINE dague_profiling_add_dictionary_keyword_f08
  END INTERFACE dague_profiling_add_dictionary_keyword_08

  INTERFACE dague_profiling_trace_f08
     SUBROUTINE dague_profiling_trace_f08(ctx, key, &
          event_id, object_id, info, ierr) &
          BIND(C, name="dague_profiling_trace_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_INT64_T
       IMPORT dague_thread_profiling_t
       IMPLICIT NONE
       TYPE(dague_thread_profiling_t)              :: ctx
       INTEGER(KIND=c_int), VALUE, INTENT(IN)      :: key
       INTEGER(KIND=c_int64_t), VALUE, INTENT(IN)  :: event_id
       INTEGER(KIND=c_int), VALUE, INTENT(IN)      :: object_id
       TYPE(c_ptr), INTENT(IN)                     :: info
       INTEGER(KIND=c_int), INTENT(OUT)            :: ierr
     END SUBROUTINE dague_profiling_trace_f08
  END INTERFACE dague_profiling_trace_f08

CONTAINS

  SUBROUTINE dague_profiling_init(hdr_id, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    CHARACTER(KIND=c_char), INTENT(IN)         :: hdr_id(*)
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_init_f08(hdr_id, c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_init
     
  SUBROUTINE dague_profiling_fini(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_fini_f08(c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_fini

  SUBROUTINE dague_profiling_reset(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_reset_f08(c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_reset

  SUBROUTINE dague_profiling_dump(fname, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    CHARACTER(KIND=c_char), INTENT(IN) :: fname(*)
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_dump_f08(fname, c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_dump

  FUNCTION dague_profiling_thread_init(length, id_name)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
    IMPLICIT NONE
    INTEGER(KIND=C_SIZE_T), VALUE, INTENT(IN) :: length
    CHARACTER(KIND=c_char), INTENT(IN)        :: id_name(*)
    TYPE(dague_thread_profiling_t)            :: dague_profiling_thread_init

    dague_profiling_thread_init = dague_profiling_thread_init_f08(length, id_name)
  END FUNCTION dague_profiling_thread_init

  SUBROUTINE dague_profiling_add_dictionary_keyword(key_name, &
       attr, info_length, conv_code, key_start, key_end, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
    IMPLICIT NONE
    CHARACTER(KIND=c_char), INTENT(IN)         :: key_name(*)
    CHARACTER(KIND=c_char), INTENT(IN)         :: attr(*)
    INTEGER(KIND=C_SIZE_T), VALUE, INTENT(IN)  :: info_length
    CHARACTER(KIND=c_char), INTENT(IN)         :: conv_code(*)
    INTEGER(KIND=c_int), INTENT(OUT)           :: key_start
    INTEGER(KIND=c_int), INTENT(OUT)           :: key_end
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_add_dictionary_keyword_f08(key_name, attr, info_length, &
         conv_code, key_start, key_end, c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_add_dictionary_keyword

  SUBROUTINE dague_profiling_trace(ctx, key, &
       event_id, object_id, info, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_INT64_T
    IMPLICIT NONE
    TYPE(dague_thread_profiling_t)              :: ctx
    INTEGER(KIND=c_int), VALUE, INTENT(IN)      :: key
    INTEGER(KIND=c_int64_t), VALUE, INTENT(IN)  :: event_id
    INTEGER(KIND=c_int), VALUE, INTENT(IN)      :: object_id
    TYPE(c_ptr), INTENT(IN)                     :: info
    INTEGER(KIND=c_int), OPTIONAL, INTENT(OUT) :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profiling_trace_f08(ctx, key, event_id, object_id, info, c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profiling_trace

end module dague_profile_f08_interfaces
