! -*- f90 -*-
! Copyright (c) 2013-2015 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! $COPYRIGHT$

module parsec_profile_f08_interfaces

  use, intrinsic :: ISO_C_BINDING

  type, BIND(C) :: parsec_profile_taskpool_t
     TYPE(C_PTR) :: TASKPOOL
  end type parsec_profile_taskpool_t

  INTERFACE parsec_profile_init_f08
     FUNCTION parsec_profile_init_f08() &
          BIND(C, name="parsec_profiling_init")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT) :: parsec_profile_init_f08
     END FUNCTION parsec_profile_init_f08
  END INTERFACE parsec_profile_init_f08

  INTERFACE parsec_profile_fini_f08
     FUNCTION parsec_profile_fini_f08() &
          BIND(C, name="parsec_profiling_fini")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT) :: parsec_profile_fini_f08
     END FUNCTION parsec_profile_fini_f08
  END INTERFACE parsec_profile_fini_f08

  INTERFACE parsec_profile_reset_f08
     FUNCTION parsec_profile_reset_f08() &
          BIND(C, name="parsec_profiling_reset")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT) :: parsec_profile_reset_f08
     END FUNCTION parsec_profile_reset_f08
  END INTERFACE parsec_profile_reset_f08

  INTERFACE parsec_profile_dump_f08
     FUNCTION parsec_profile_dump_f08() &
          BIND(C, name="parsec_profiling_dbp_dump")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT
       IMPLICIT NONE
       INTEGER(KIND=C_INT) :: parsec_profile_dump_f08
     END FUNCTION parsec_profile_dump_f08
  END INTERFACE parsec_profile_dump_f08

  INTERFACE parsec_profile_thread_init_f08
     FUNCTION parsec_profile_thread_init_f08(length, id_name, ierr) &
          BIND(C, name="parsec_profile_thread_init_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_CHAR, C_SIZE_T, C_PTR, C_INT
       IMPORT parsec_profile_taskpool_t
       IMPLICIT NONE
       INTEGER(KIND=C_SIZE_T), INTENT(IN),VALUE     :: length
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: id_name(*)
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
       TYPE(parsec_profile_taskpool_t)              :: parsec_profile_thread_init_f08
     END FUNCTION parsec_profile_thread_init_f08
  END INTERFACE parsec_profile_thread_init_f08

  INTERFACE parsec_profile_add_dictionary_keyword_08
     SUBROUTINE parsec_profile_add_dictionary_keyword_f08(key_name, &
          attr, info_length, conv_code, key_start, key_end, ierr) &
          BIND(C, name="parsec_profile_add_dictionary_keyword_f08")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T
       IMPLICIT NONE
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: key_name(*)
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: attr(*)
       INTEGER(KIND=C_SIZE_T), INTENT(IN), VALUE    :: info_length
       CHARACTER(KIND=C_CHAR), INTENT(IN)           :: conv_code(*)
       INTEGER(KIND=C_INT), INTENT(OUT)             :: key_start
       INTEGER(KIND=C_INT), INTENT(OUT)             :: key_end
       INTEGER(KIND=C_INT), INTENT(OUT)             :: ierr
     END SUBROUTINE parsec_profile_add_dictionary_keyword_f08
  END INTERFACE parsec_profile_add_dictionary_keyword_08

  INTERFACE parsec_profile_trace_f08
     FUNCTION parsec_profile_trace_f08(ctx, key, &
          event_id, object_id, info, flags) &
          BIND(C, name="parsec_profiling_trace_flags")
       USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_INT16_T, C_PTR, C_INT64_T
       IMPORT parsec_profile_taskpool_t
       IMPLICIT NONE
       TYPE(parsec_profile_taskpool_t), VALUE        :: ctx
       INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: key
       INTEGER(KIND=C_INT64_T), INTENT(IN), VALUE   :: event_id
       INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: object_id
       TYPE(c_ptr), INTENT(IN), VALUE               :: info
       INTEGER(KIND=C_INT16_T), INTENT(IN), VALUE   :: flags
       INTEGER(KIND=C_INT)                          :: parsec_profile_trace_f08
     END FUNCTION parsec_profile_trace_f08
  END INTERFACE parsec_profile_trace_f08

  INTERFACE parsec_profile_enable_f08
     SUBROUTINE parsec_profile_enable_f08() &
          BIND(C, name="parsec_profiling_enable")
       IMPLICIT NONE
     END SUBROUTINE parsec_profile_enable_f08
  END INTERFACE parsec_profile_enable_f08

  INTERFACE parsec_profile_disable_f08
     SUBROUTINE parsec_profile_disable_f08() &
          BIND(C, name="parsec_profiling_disable")
       IMPLICIT NONE
     END SUBROUTINE parsec_profile_disable_f08
  END INTERFACE parsec_profile_disable_f08

  INTERFACE parsec_profile_start_f08
    FUNCTION parsec_profile_start_f08(f_name, hr_info) &
          BIND(C, name="parsec_profiling_dbp_start")
      USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
      IMPLICIT NONE
      CHARACTER(KIND=C_CHAR), INTENT(IN)           :: f_name(*)
      CHARACTER(KIND=C_CHAR), INTENT(IN)           :: hr_info(*)
      INTEGER(KIND=C_INT)                          :: parsec_profile_start_f08
    END FUNCTION parsec_profile_start_f08
  END INTERFACE parsec_profile_start_f08

CONTAINS

  SUBROUTINE parsec_profile_init(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err

    c_err = parsec_profile_init_f08()
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE parsec_profile_init

  SUBROUTINE parsec_profile_fini(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    c_err = parsec_profile_fini_f08()
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE parsec_profile_fini

  SUBROUTINE parsec_profile_reset(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr
    INTEGER(KIND=c_int)                        :: c_err

    c_err = parsec_profile_reset_f08()
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE parsec_profile_reset

  SUBROUTINE parsec_profile_dump(ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR
    IMPLICIT NONE
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err

    c_err = parsec_profile_dump_f08()
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE parsec_profile_dump

  FUNCTION parsec_profile_thread_init(length, id_name, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT, C_CHAR, C_SIZE_T, C_PTR
    IMPLICIT NONE
    INTEGER*8, INTENT(IN)                     :: length
    CHARACTER(*), INTENT(IN)                  :: id_name
    INTEGER, OPTIONAL, INTENT(OUT)            :: ierr
    TYPE(parsec_profile_taskpool_t)           :: parsec_profile_thread_init

    INTEGER(KIND=C_SIZE_T)                    :: c_length
    CHARACTER(KIND=C_CHAR), POINTER           :: c_id_name(:)
    INTEGER(KIND=C_INT)                       :: c_ierr
    INTEGER                                   :: i

    c_length = length
    ALLOCATE(c_id_name(LEN_TRIM(id_name)+1))
    c_id_name(:) = (/ (id_name(i:i), i = 1, LEN_TRIM(id_name)), c_null_char /)
    parsec_profile_thread_init = parsec_profile_thread_init_f08(c_length, c_id_name, c_ierr)
    DEALLOCATE(c_id_name)
    if(present(ierr)) ierr = c_ierr;
  END FUNCTION parsec_profile_thread_init

  SUBROUTINE parsec_profile_add_dictionary_keyword(key_name, &
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
    call parsec_profile_add_dictionary_keyword_f08(c_key_name, c_attr, c_info_length, &
                                                  c_conv_code, c_key_start, c_key_end, c_err)
    key_start = c_key_start
    key_end = c_key_end
    if(present(ierr)) ierr = c_err;
    DEALLOCATE(c_key_name)
    DEALLOCATE(c_attr)
    DEALLOCATE(c_conv_code)
  END SUBROUTINE parsec_profile_add_dictionary_keyword

  SUBROUTINE parsec_profile_trace(ctx, key, &
       event_id, object_id, info, ierr)
    USE, intrinsic :: ISO_C_BINDING, only : C_INT16_T, C_INT, C_INT64_T
    IMPLICIT NONE
    TYPE(parsec_profile_taskpool_t)            :: ctx
    INTEGER, INTENT(IN)                        :: key
    INTEGER*8, INTENT(IN)                      :: event_id
    INTEGER, INTENT(IN)                        :: object_id
    TYPE(c_ptr), INTENT(IN)                    :: info
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_err
    INTEGER(KIND=c_int64_t)                    :: c_event_id
    INTEGER(KIND=C_INT16_T)                    :: zero

    c_event_id = event_id
    zero = 0
    c_err = parsec_profile_trace_f08(ctx, key, event_id, object_id, info, zero)
    if(present(ierr)) ierr = c_err;
  END SUBROUTINE parsec_profile_trace

  SUBROUTINE parsec_profile_enable()
    IMPLICIT NONE

    call parsec_profile_enable_f08()
  END SUBROUTINE parsec_profile_enable

  SUBROUTINE parsec_profile_disable()
    IMPLICIT NONE

    call parsec_profile_disable_f08()
  END SUBROUTINE parsec_profile_disable

  SUBROUTINE parsec_profile_start(fname, hr_info, ierr)
    IMPLICIT NONE
    CHARACTER(*), INTENT(IN)                   :: fname
    CHARACTER(*), INTENT(IN)                   :: hr_info
    CHARACTER(KIND=C_CHAR), ALLOCATABLE        :: c_fname(:)
    CHARACTER(KIND=C_CHAR), ALLOCATABLE        :: c_hr_info(:)
    INTEGER, OPTIONAL, INTENT(OUT)             :: ierr

    INTEGER(KIND=c_int)                        :: c_ierr
    INTEGER i

    ALLOCATE(c_fname(LEN_TRIM(fname)+1))
    c_fname(:) = (/ (fname(i:i), i = 1, LEN_TRIM(fname)), c_null_char /)

    ALLOCATE(c_hr_info(LEN_TRIM(hr_info)+1))
    c_hr_info(:) = (/ (hr_info(i:i), i = 1, LEN_TRIM(hr_info)), c_null_char /)

    c_ierr = parsec_profile_start_f08(c_fname, c_hr_info)
    if(present(ierr)) ierr = c_ierr;
    DEALLOCATE(c_fname)
    DEALLOCATE(c_hr_info)
  END SUBROUTINE parsec_profile_start

end module parsec_profile_f08_interfaces
