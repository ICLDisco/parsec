PROGRAM GENERATE_F

  use, INTRINSIC :: ISO_C_BINDING, only : c_int, C_NULL_PTR
  use dague_profile_f08_interfaces

  integer BLOCK, N, ret
  parameter (BLOCK=10, N=100)
  integer(8) info_length, prof_length, event_id

  type(dague_profile_handle_t) :: prof_handle

  info_length = 0
  call dague_profile_init(ierr)

  call dague_profile_add_dictionary_keyword("key 1", "NULL attr", &
       info_length, "NO CONV", k1_start, k1_end, ierr)
  call dague_profile_add_dictionary_keyword("key 2", "NULL attr", &
       info_length, "NO CONV", k2_start, k2_end, ierr)
  call dague_profile_add_dictionary_keyword("key 3", "NULL attr", &
       info_length, "NO CONV", k3_start, k3_end, ierr)

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
     call dague_profile_trace(prof_handle, k1_start, event_id, 1, &
          C_NULL_PTR, ierr)
     call dague_profile_trace(prof_handle, k1_end, event_id, 1, &
          C_NULL_PTR, ierr)
  end do

  call dague_profile_dump(ierr)

  call dague_profile_fini(ierr)

  call exit(ret)
END

