####################################
# DBPreader Python interface
# run 'python setup.py build_ext --inplace' to compile
# import py_dbpreader to use
#
# This is verified to work with Python 2.4.3 and above, compiled with Cython 0.16rc0
# However, it is recommended that Python 2.5 or greater is used to build and run,
# as certain sorting functions are not available until that version of Python.
#
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.

import sys
from libc.stdlib cimport malloc, free
from parsec_profile import * # the pure Python classes
import gc

# Cython-y code

# this is the public Python interface function. call it.
cpdef readProfile(filenames):
   cdef char ** c_filenames = stringListToCStrings(filenames)
   cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
#   cdef dbp_multifile_reader_t * dbp = dbp_reader_open_default_files()
   profile = multifile_reader(dbp_reader_nb_files(dbp), dbp_reader_nb_dictionary_entries(dbp))
   print('PYX: finished reading')
   cdef dbp_file_t * cfile
   cdef dbp_dictionary_t * cdict

   profile.worldsize = dbp_reader_worldsize(dbp)

   print('here we are...')

   # create dictionary first, for later use while making Events
   for index in range(profile.nb_dict_entries):
      cdict = dbp_reader_get_dictionary(dbp, index)
      entry = dbpDictEntry(index, dbp_dictionary_attributes(cdict))
      profile.dictionary[dbp_dictionary_name(cdict)] = entry

   # convert c to py
   for ifd in range(profile.nb_files):
      print('file ' + str(ifd) + ' of ' + str(profile.nb_files)) 
      cfile = dbp_reader_get_file(dbp, ifd)
      pfile = dbpFile(profile, dbp_file_hr_id(cfile), dbp_file_get_name(cfile), dbp_file_get_rank(cfile))
      for index in range(dbp_file_nb_infos(cfile)):
         print('info ' + str(index))
         cinfo = dbp_file_get_info(cfile, index)
         key = dbp_info_get_key(cinfo)
         value = dbp_info_get_value(cinfo)
         pfile.infos.append(dbpInfo(key, value))
      for thread_num in range(dbp_file_nb_threads(cfile)):
         print('thread ' + str(thread_num) + ' of ' + str(dbp_file_nb_threads(cfile)))
         new_thr = makeDbpThread(profile, dbp, cfile, thread_num, pfile)
         pfile.threads.append(new_thr)
      profile.files.append(pfile)

   dbp_reader_close_files(dbp) # does nothing as of 2013-04-21
#   dbp_reader_dispose_reader(dbp)
#   free(c_filenames)

   return profile

# helper function for readProfile
cdef char** stringListToCStrings(strings):
   cdef char ** c_argv
   bytes_strings = [bytes(x) for x in strings]
   c_argv = <char**>malloc(sizeof(char*) * len(bytes_strings)) 
   if c_argv is NULL:
      raise MemoryError()
   try:
      for idx, s in enumerate(bytes_strings):
         c_argv[idx] = s
   except:
      print("exception caught while converting to c strings")
#      free(c_argv)
   return c_argv

# you can't call this. it will be called for you. call readProfile()
cdef makeDbpThread(reader, dbp_multifile_reader_t * dbp, dbp_file_t * cfile, int index, pfile):
   cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, index)
   print('got cthread ')
   cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
   print('got iterator')
   cdef dbp_event_iterator_t * it_e = NULL
   cdef dbp_event_t * event_s = dbp_iterator_current(it_s)
   print('got event')
   cdef dbp_event_t * event_e = NULL
   cdef dague_time_t reader_start = dbp_reader_min_date(dbp)
   print('got min date')
   cdef unsigned long long start = 0
   cdef unsigned long long end = 0
   cdef void * cinfo = NULL
   cdef papi_exec_info_t * cast_exec_info = NULL
   cdef select_info_t * cast_select_info = NULL

   print('make the thread ' + str(index))

   thread = dbpThread(pfile, index)
   if thread.id + 1 > reader.thread_count:
      reader.thread_count = thread.id + 1

   print('one')

   while event_s != NULL:
      if KEY_IS_START( dbp_event_get_key(event_s) ):
         it_e = dbp_iterator_find_matching_event_all_threads(it_s)
         if it_e != NULL:
            event_e = dbp_iterator_current(it_e)
            if event_e != NULL:
               start = diff_time(reader_start, dbp_event_get_timestamp(event_s))
               end = diff_time(reader_start, dbp_event_get_timestamp(event_e))
               event_key = int(dbp_event_get_key(event_s)) / 2 # to match dictionary
               event_flags = dbp_event_get_flags(event_s)
               event_handle_id = int(dbp_event_get_handle_id(event_s))
               event_id = int(dbp_event_get_event_id(event_s))
               print("{} {} {} {} {} {} {}".format(thread, event_key, event_flags, event_handle_id, event_id, start, end))
               event = dbpEvent(thread,
                                event_key,
                                event_flags,
                                event_handle_id,
                                event_id,
                                start, end)
               print('next?')
               while len(reader.handle_counts) < (event_handle_id + 1):
                  reader.handle_counts.extend([0])
               reader.handle_counts[event_handle_id] += 1
               #####################################
               # not all events have info
               # also, not all events have the same info.
               # so this is where users must add code to translate
               # their own info objects
               cinfo = dbp_event_get_info(event_e)
               if cinfo != NULL:
                  if (reader.dictionary.get('PINS_EXEC', None) and
                      event_key == reader.dictionary['PINS_EXEC'].id):
                     cast_exec_info = <papi_exec_info_t *>cinfo
                     values_len = int(cast_exec_info.values_len)
#                     gc.collect()
                     values_lst = [cast_exec_info.values[x] for x in range(values_len)]
                     kern_name = str(cast_exec_info.kernel_name)
#                     gc.collect()
                     event.info = dbp_Exec_EventInfo(
                             cast_exec_info.kernel_type,
                             kern_name,
                             cast_exec_info.vp_id,
                             cast_exec_info.th_id,
                             values_lst)
                  elif (reader.dictionary.get('PINS_SELECT', None) and
                        event_key == reader.dictionary['PINS_SELECT'].id):
                     cast_select_info = <select_info_t *>cinfo
                     values_len = int(cast_select_info.values_len)
#                     gc.collect()
                     values_lst = [cast_select_info.values[x] for x in range(values_len)]
#                     gc.collect()
                     event.info = dbp_Select_EventInfo(
                                cast_select_info.kernel_type,
                                cast_select_info.vp_id,
                                cast_select_info.th_id,
                                cast_select_info.victim_vp_id,
                                cast_select_info.victim_th_id,
                                cast_select_info.exec_context,
                                values_lst)
                  # elif event.key == reader.dictionary['<SOME OTHER TYPE WITH INFO>].id:
                     # event.info = <write a function and a Python type to translate>
                  else:
   #                  print(event.key)
   #                  print([str(k) + ' ' + str(v.id) for k, v in reader.dictionary.items()])
                     unused = None

               thread.events.append(event)
            dbp_iterator_delete(it_e)
            it_e = NULL
      dbp_iterator_next(it_s)
      event_s = dbp_iterator_current(it_s)
   print('PYX: returning thread')
   dbp_iterator_delete(it_s)
   it_s = NULL
   return thread

