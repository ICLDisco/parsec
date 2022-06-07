# pbt2ptt python definition file
cdef extern from "parsec/parsec_config.h":
    pass

cdef extern from "parsec/parsec_binary_profile.h":
   ctypedef struct parsec_profiling_stream_t:
      pass

   int KEY_IS_START(int key)
   int KEY_IS_END(int key)
   int BASE_KEY(int key)

cdef extern from "stdint.h":
    ctypedef int uint64_t

cdef extern from "dbpreader.h":
   ctypedef struct dbp_info_t:
      pass

   ctypedef struct dbp_thread_t:
      pass

   ctypedef struct dbp_file_t:
      pass

   ctypedef struct dbp_dictionary_t:
      pass

   ctypedef struct dbp_multifile_reader_t:
      pass

   ctypedef struct dbp_event_t:
      pass

   ctypedef struct dbp_event_iterator_t:
      pass

   char * dbp_info_get_key(dbp_info_t * info)
   char * dbp_info_get_value(dbp_info_t * info)

   dbp_multifile_reader_t* dbp_reader_open_files(int nbfiles, char * files[])
   int dbp_reader_nb_files(dbp_multifile_reader_t * dbp)
   int dbp_reader_nb_dictionary_entries(dbp_multifile_reader_t * dbp)
   int dbp_reader_last_error(const dbp_multifile_reader_t *dbp)
   void dbp_reader_close_files(dbp_multifile_reader_t * dbp)
   void dbp_reader_destruct(dbp_multifile_reader_t * dbp)

   dbp_dictionary_t * dbp_file_get_dictionary(dbp_file_t * file, int did)
   dbp_dictionary_t * dbp_reader_get_dictionary(dbp_multifile_reader_t * dbp, int did)
   int dbp_file_translate_local_dico_to_global(dbp_file_t *file, int lid)
   char * dbp_dictionary_name(dbp_dictionary_t * dico)
   char * dbp_dictionary_attributes(dbp_dictionary_t * dico)
   char * dbp_dictionary_convertor(dbp_dictionary_t * dico)
   int dbp_dictionary_keylen(dbp_dictionary_t * dico)

   dbp_file_t *dbp_reader_get_file(dbp_multifile_reader_t *dbp, int fid)

   char *dbp_file_hr_id(dbp_file_t *file)
   char *dbp_file_get_name(dbp_file_t *file)
   int dbp_file_get_rank(dbp_file_t *file)
   int dbp_file_nb_threads(dbp_file_t *file)
   int dbp_file_nb_infos(dbp_file_t *file)
   int dbp_file_nb_dictionary_entries(const dbp_file_t *file)
   int dbp_file_error(const dbp_file_t *file)
   dbp_info_t *dbp_file_get_info(dbp_file_t *file, int iid)

   dbp_thread_t *dbp_file_get_thread(dbp_file_t *file, int tid)

   int dbp_thread_nb_events(dbp_thread_t *th)
   int dbp_thread_nb_infos(dbp_thread_t *th)
   char * dbp_thread_get_hr_id(dbp_thread_t *th)
   dbp_info_t *dbp_thread_get_info(dbp_thread_t *th, int iid)

   dbp_event_iterator_t *dbp_iterator_new_from_thread(dbp_thread_t *th)
   dbp_event_iterator_t *dbp_iterator_new_from_iterator(dbp_event_iterator_t *it)
   dbp_event_t *dbp_iterator_current(dbp_event_iterator_t *it)
   dbp_event_t *dbp_iterator_first(dbp_event_iterator_t *it)
   dbp_event_t *dbp_iterator_next(dbp_event_iterator_t *it)
   void dbp_iterator_delete(dbp_event_iterator_t *it)
   int dbp_iterator_move_to_matching_event(dbp_event_iterator_t *pos, dbp_event_t *ref)
   dbp_event_iterator_t *dbp_iterator_find_matching_event_all_threads(dbp_event_iterator_t *pos)
   dbp_thread_t *dbp_iterator_thread(dbp_event_iterator_t *it)

   int dbp_event_get_key(dbp_event_t *e)
   int dbp_event_get_flags(dbp_event_t *e)
   long long int dbp_event_get_event_id(dbp_event_t *e)
   int dbp_event_get_taskpool_id(dbp_event_t *e)
   long long int dbp_event_get_timestamp(dbp_event_t *e)
   char *dbp_event_get_info(dbp_event_t *e)
   int   dbp_event_info_len(dbp_event_t *e, dbp_multifile_reader_t *dbp)

   # DEBUG
   void dbp_file_print(dbp_file_t* file)
