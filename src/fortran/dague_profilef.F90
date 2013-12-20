! -*- f90 -*-
! Copyright (c) 2013      The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module dague_profile_f08_interfaces

  use, intrinsic :: ISO_C_BINDING

  type, BIND(C) :: dague_profile_handle_t
     TYPE(C_PTR) :: HANDLE
  end type dague_profile_handle_t

  INTERFACE dague_profile_init_f08
     SUBROUTINE dague_profile_init_f08(hdr_id, ierr) &
          BIND(C, name="dague_profile_init_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
       IMPLICIT NONE
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: hdr_id(*)
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_init_f08
  END INTERFACE dague_profile_init_f08

  INTERFACE dague_profile_fini_f08
     SUBROUTINE dague_profile_fini_f08(ierr) &
          BIND(C, name="dague_profile_fini_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_fini_f08
  END INTERFACE dague_profile_fini_f08

  INTERFACE dague_profile_reset_f08
     SUBROUTINE dague_profile_reset_f08(ierr) &
          BIND(C, name="dague_profile_reset_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_reset_f08
  END INTERFACE dague_profile_reset_f08

  INTERFACE dague_profile_dump_f08
     SUBROUTINE dague_profile_dump_f08(fname, ierr) &
          BIND(C, name="dague_profile_dump_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
       IMPLICIT NONE
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: fname(*)
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_dump_f08
  END INTERFACE dague_profile_dump_f08

  INTERFACE dague_profile_thread_init_f08
     FUNCTION dague_profile_thread_init_f08(length, id_name) &
          BIND(C, name="dague_profile_thread_init_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_CHAR, C_SIZE_T, C_PTR
       IMPORT dague_profile_handle_t
       IMPLICIT NONE
       INTEGER(KIND=C_SIZE_T), INTENT(IN),VALUE     :: length
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: id_name(*)
       TYPE(dague_profile_handle_t)                 :: dague_profile_thread_init_f08
     END FUNCTION dague_profile_thread_init_f08
  END INTERFACE dague_profile_thread_init_f08

  INTERFACE dague_profile_add_dictionary_keyword_08
     SUBROUTINE dague_profile_add_dictionary_keyword_f08(key_name, &
          attr, info_length, conv_code, key_start, key_end, ierr) &
          BIND(C, name="dague_profile_add_dictionary_keyword_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
       IMPLICIT NONE
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: key_name(*)
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: attr(*)
       INTEGER(KIND=C_SIZE_T), INTENT(IN),VALUE     :: info_length
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: conv_code(*)
       INTEGER(KIND=C_INT), INTENT(OUT)             :: key_start
       INTEGER(KIND=C_INT), INTENT(OUT)             :: key_end
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_add_dictionary_keyword_f08
  END INTERFACE dague_profile_add_dictionary_keyword_08

  INTERFACE dague_profile_trace_f08
     SUBROUTINE dague_profile_trace_f08(ctx, key, &
          event_id, object_id, info, ierr) &
          BIND(C, name="dague_profile_trace_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_INT64_T
       IMPORT dague_profile_handle_t
       IMPLICIT NONE
       TYPE(dague_profile_handle_t)                 :: ctx
       INTEGER(KIND=C_INT), INTENT(IN),VALUE        :: key
       INTEGER(KIND=C_INT64_T), INTENT(IN),VALUE    :: event_id
       INTEGER(KIND=C_INT), INTENT(IN),VALUE        :: object_id
       TYPE(c_ptr), INTENT(IN)                      :: info
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE dague_profile_trace_f08
  END INTERFACE dague_profile_trace_f08

  INTERFACE dague_profile_enable_f08
     SUBROUTINE dague_profile_enable_f08() &
          BIND(C, name="dague_profiling_enable")
       IMPLICIT NONE
     END SUBROUTINE dague_profile_enable_f08
  END INTERFACE dague_profile_enable_f08

  INTERFACE dague_profile_disable_f08
     SUBROUTINE dague_profile_disable_f08() &
          BIND(C, name="dague_profiling_disable")
       IMPLICIT NONE
     END SUBROUTINE dague_profile_disable_f08
  END INTERFACE dague_profile_disable_f08

  INTERFACE dague_profile_start_f08
    FUNCTION dague_profile_start_f08() &
          BIND(C, name="dague_profiling_start")
      USE, intrinsic :: ISO_C_BINDING, only : C_INT
      IMPLICIT NONE
      INTEGER(KIND=C_INT)                          :: dague_profile_start_f08
    END FUNCTION dague_profile_start_f08
  END INTERFACE dague_profile_start_f08

CONTAINS

  SUBROUTINE dague_profile_init(hdr_id, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    CHARACTER(*), INTENT(IN)                   :: hdr_id
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err
    CHARACTER(KIND=C_CHAR), ALLOCATABLE        :: c_hdr_id(:)
    INTEGER                                    :: i

    ALLOCATE(c_hdr_id(LEN_TRIM(hdr_id)+1))
    c_hdr_id(:) = (/ (hdr_id(i:i), i = 1, LEN_TRIM(hdr_id)), c_null_char /)
    call dague_profile_init_f08(c_hdr_id, c_err)
    if(present(ierr)) ierr = c_err;
    DEALLOCATE(c_hdr_id)
  END SUBROUTINE dague_profile_init

  SUBROUTINE dague_profile_fini(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profile_fini_f08(c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profile_fini

  SUBROUTINE dague_profile_reset(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    call dague_profile_reset_f08(c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profile_reset

  SUBROUTINE dague_profile_dump(fname, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    CHARACTER(*), INTENT(IN)                   :: fname
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err
    CHARACTER(KIND=C_CHAR), ALLOCATABLE        :: c_fname(:)
    INTEGER                                    :: i

    ALLOCATE(c_fname(LEN_TRIM(fname)+1))
    c_fname(:) = (/ (fname(i:i), i = 1, LEN_TRIM(fname)), c_null_char /)
    call dague_profile_dump_f08(c_fname, c_err)
    if(present(ierr)) ierr = c_err;
    DEALLOCATE(c_fname)
  END SUBROUTINE dague_profile_dump

  FUNCTION dague_profile_thread_init(length, id_name)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T, C_PTR
    IMPLICIT NONE
    INTEGER*8, INTENT(IN)                     :: length
    CHARACTER(*), INTENT(IN)                  :: id_name
    TYPE(dague_profile_handle_t)              :: dague_profile_thread_init

    INTEGER(KIND=C_SIZE_T)                    :: c_length
    CHARACTER(KIND=C_CHAR), POINTER           :: c_id_name(:)
    INTEGER                                   :: i

    c_length = length
    ALLOCATE(c_id_name(LEN_TRIM(id_name)+1))
    c_id_name(:) = (/ (id_name(i:i), i = 1, LEN_TRIM(id_name)), c_null_char /)
    dague_profile_thread_init = dague_profile_thread_init_f08(c_length, c_id_name)
    DEALLOCATE(c_id_name)
  END FUNCTION dague_profile_thread_init

  SUBROUTINE dague_profile_add_dictionary_keyword(key_name, &
       attr, info_length, conv_code, key_start, key_end, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
    IMPLICIT NONE
    CHARACTER(*), INTENT(IN)                   :: key_name
    CHARACTER(*), INTENT(IN)                   :: attr
    INTEGER*8, INTENT(IN)                      :: info_length
    CHARACTER(*), INTENT(IN)                   :: conv_code
    INTEGER, INTENT(OUT)                       :: key_start
    INTEGER, INTENT(OUT)                       :: key_end
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err, c_key_start, c_key_end
    INTEGER(KIND=C_SIZE_T)                     :: c_info_length
    CHARACTER(KIND=C_CHAR), POINTER            :: c_key_name(:)
    CHARACTER(KIND=C_CHAR), POINTER            :: c_attr(:)
    CHARACTER(KIND=C_CHAR), POINTER            :: c_conv_code(:)
    INTEGER                                    :: i

    c_info_length = info_length
    ALLOCATE(c_key_name(LEN_TRIM(key_name)+1))
    c_key_name(:) = (/ (key_name(i:i), i = 1, LEN_TRIM(key_name)), c_null_char /)
    ALLOCATE(c_attr(LEN_TRIM(attr)+1))
    c_attr(:) = (/ (attr(i:i), i = 1, LEN_TRIM(attr)), c_null_char /)
    ALLOCATE(c_conv_code(LEN_TRIM(conv_code)+1))
    c_conv_code(:) = (/ (conv_code(i:i), i = 1, LEN_TRIM(conv_code)), c_null_char /)
    call dague_profile_add_dictionary_keyword_f08(c_key_name, c_attr, c_info_length, &
                                                  c_conv_code, c_key_start, c_key_end, c_err)
    key_start = c_key_start
    key_end = c_key_end
    if(present(ierr)) ierr = c_err;
    DEALLOCATE(c_key_name)
    DEALLOCATE(c_attr)
    DEALLOCATE(c_conv_code)
  END SUBROUTINE dague_profile_add_dictionary_keyword

  SUBROUTINE dague_profile_trace(ctx, key, &
       event_id, object_id, info, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_INT64_T
    IMPLICIT NONE
    TYPE(dague_profile_handle_t)               :: ctx
    INTEGER, INTENT(IN)                        :: key
    INTEGER*8, INTENT(IN)                      :: event_id
    INTEGER, INTENT(IN)                        :: object_id
    TYPE(c_ptr), INTENT(IN)                    :: info
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err
    INTEGER(KIND=c_int64_t)                    :: c_event_id

    c_event_id = event_id
    call dague_profile_trace_f08(ctx, key, event_id, object_id, info, c_err)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE dague_profile_trace

  SUBROUTINE dague_profile_enable()
    IMPLICIT NONE

    call dague_profile_enable_f08()
  END SUBROUTINE dague_profile_enable

  SUBROUTINE dague_profile_disable()
    IMPLICIT NONE

    call dague_profile_disable_f08()
  END SUBROUTINE dague_profile_disable

  SUBROUTINE dague_profile_start(ierr)
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_ierr
    c_ierr = dague_profile_start_f08();
    if(present(ierr)) ierr = c_ierr;
  END SUBROUTINE dague_profile_start

end module dague_profile_f08_interfaces
