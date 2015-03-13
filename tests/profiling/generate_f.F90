PROGRAM GENERATE_F

  use, INTRINSIC :: ISO_C_BINDING, only : c_int, C_NULL_PTR
  use dague_profile_f08_interfaces

  type dfi_type
     real*8:: d
     real*4:: f
     integer*4:: i
  end type dfi_type

  type dd_type
     real*8:: d1
     real*8:: d2
  end type dd_type

  integer BLOCK, N, ret
  parameter (BLOCK=10, N=100)
  integer(8) info_length, prof_length, event_id
  type(dfi_type), target :: dfi
  type(dd_type), target  :: dd

  type(dague_profile_handle_t) :: prof_handle

  call dague_profile_init(ierr)

  info_length = SIZEOF(dfi)
  call dague_profile_add_dictionary_keyword("key 1", "NULL attr", &
       info_length, "double(%d):float(%f):int(%d)", k1_start, k1_end, ierr)
  call dague_profile_add_dictionary_keyword("key 2", "NULL attr", &
       info_length, "double(%d):float(%f):int(%d)", k2_start, k2_end, ierr)
  info_length = SIZEOF(dd)
  call dague_profile_add_dictionary_keyword("key 3", "NULL attr", &
       info_length, "double1(%d):double2(%d)", k3_start, k3_end, ierr)

  prof_length = 1024 * 1024
  prof_handle = dague_profile_thread_init(prof_length, "thread 1", ierr)
  if(0.eq.ierr) then
     write(*,*) 'Call to dague_profile_thread_init should have FAILED'
  endif

  call dague_profile_start("myfile", "MYKEY", ierr)

  prof_handle = dague_profile_thread_init(prof_length, "thread 1", ierr)
  if(0.ne.ierr) then
     write(*,*) 'Call to dague_profile_thread_init FAILED'
  endif

  event_id = 1
  do i = 1, 1000, 1
     dfi%d = i * 1.0d0
     dfi%f = i * 1.0d0
     dfi%i = i
     call dague_profile_trace(prof_handle, k1_start, event_id, 1, &
          C_LOC(dfi), ierr)
     dfi%f = i * 2.0d0
     call dague_profile_trace(prof_handle, k1_end, event_id, 1, &
          C_LOC(dfi), ierr)
     dd%d1 = i * 1.0d0
     dd%d2 = i * 2.0d0
     call dague_profile_trace(prof_handle, k3_start, event_id, 1, &
          C_NULL_PTR, ierr)
     dd%d1 = i * 4.0d0
     dd%d2 = i * 8.0d0
     call dague_profile_trace(prof_handle, k3_end, event_id, 1, &
          C_LOC(dd), ierr)
  end do

  call dague_profile_dump(ierr)

  call dague_profile_fini(ierr)

  call exit(ret)
END

