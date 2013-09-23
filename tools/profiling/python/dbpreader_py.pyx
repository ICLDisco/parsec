####################################
# DBPreader Python interface
# run 'python setup.py build_ext --inplace' to compile
# import dbpreader_py to use
#
# This is verified to work with Python 2.4.3 and above, compiled with Cython 0.16rc0
# However, it is recommended that Python 2.5 or greater is used to build and run,
# as certain sorting functions are not available until that version of Python.
#
# Be SURE to build this against the same version of Python as you have built Cython itself.
# Contrasting versions will likely lead to odd errors about Unicode functions.

import sys
from libc.stdlib cimport malloc, free

# pure Python

class d_time:
   def __init__(self, seconds, nsec):
      self.sec = seconds
      self.nsec = nsec
   def diff(self, time):
      return d_time(time.sec - self.sec, time.nsec - self.nsec)
   def abs(self):
      return self.sec * 1000000000 + self.nsec

class multifile_reader:
   def __init__(self, nb_files, nb_dict_entries):
      self.nb_files = nb_files
      self.nb_dict_entries = nb_dict_entries
      self.dictionary = {}
      self.files = []

class dbpDictEntry:
   def __init__(self, id, name, attributes):
      self.id = id
      self.name = name
      self.attributes = attributes

class dbpFile:
   def __init__(self, parent, hr_id, filename, rank):
      self.parent = parent
      self.hr_id = hr_id
      self.filename = filename
      self.rank = rank
      self.infos = []
      self.threads = []
      # NOTE: maybe collect statistics on timing stuff later

class dbpInfo:
   def __init__(self, key, value):
      self.key = key
      self.value = value

class dbpThread:
   def __init__(self, parentFile, threadNumber):
      self.file = parentFile
      self.events = []
      self.id = int(threadNumber) # if it's not a number, it's wrong

class dbpEvent:
   def __init__(self, parentThread, key, flags, object_id, event_id, start, end):
      self.thread = parentThread
      self.key = key
      self.flags = flags
      self.object_id = object_id
      self.event_id = event_id
      self.start = start
      self.end = end
      self.duration = self.end - self.start
   def __str__(self):
      return 'key %d flags %d tid %d objID %s eventID %d start %d end %d duration %d' % (
              self.key, self.flags, self.thread.id, self.object_id, self.event_id, self.start, self.end, self.duration)


# Cython code

# this is the public Python interface function. 
cpdef readProfilesIntoPython(filenames):
   cdef char ** c_filenames = stringListToCStrings(filenames)
   cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
   reader = multifile_reader(dbp_reader_nb_files(dbp), dbp_reader_nb_dictionary_entries(dbp))
   cdef dbp_file_t * cfile   
   cdef dbp_dictionary_t * cdict

   # convert c to py
   for ifd in range(reader.nb_files):
      cfile = dbp_reader_get_file(dbp, ifd)
      file = dbpFile(reader, dbp_file_hr_id(cfile), dbp_file_get_name(cfile), dbp_file_get_rank(cfile))
      for inf in range(dbp_file_nb_infos(cfile)):
         file.infos.append(makeDbpInfo(cfile, inf))
      for t in range(dbp_file_nb_threads(cfile)):
         file.threads.append(makeDbpThread(dbp, cfile, t, file))

      reader.files.append(file)

   for dce in range(reader.nb_dict_entries):
      reader.dictionary[dce] = makeDbpDictEntry(dbp, dce)

   reader.worldsize = dbp_reader_worldsize(dbp)

   free(c_filenames)  
   # also, free multifile_reader and associated event buffers?
   return reader


cdef char** stringListToCStrings(strings):
   cdef char ** c_argv
   strings = [bytes(x) for x in strings]
   c_argv = <char**>malloc(sizeof(char*) * len(strings)) 
   if c_argv is NULL:
      raise MemoryError()
   try:   
      for idx, s in enumerate(strings):
         c_argv[idx] = s
   except:
      print("exception caught while converting to c strings")
      free(c_argv)
   return c_argv

cdef makeDbpInfo(dbp_file_t * cfile, int index):
   cdef dbp_info_t * cinfo = dbp_file_get_info(cfile, index)
   key = dbp_info_get_key(cinfo)
   value = dbp_info_get_value(cinfo)
   return dbpInfo(key, value)

cdef makeDbpDictEntry(dbp_multifile_reader_t * dbp, int index):
   dico = dbp_reader_get_dictionary(dbp, index)
   return dbpDictEntry(index, dbp_dictionary_name(dico), dbp_dictionary_attributes(dico))

cdef makeDbpThread(dbp_multifile_reader_t * dbp, dbp_file_t * cfile, int index, file):
   cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, index)
   cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
   cdef dbp_event_iterator_t * it_e
   cdef const dbp_event_t * event_s = dbp_iterator_current(it_s)
   cdef const dbp_event_t * event_e
   cdef dague_time_t reader_start = dbp_reader_min_date(dbp)
   cdef unsigned long long start, end

   thread = dbpThread(file, index)

   while event_s is not NULL:
      if KEY_IS_START( dbp_event_get_key(event_s) ):
         it_e = dbp_iterator_find_matching_event_all_threads(it_s, 0)
         if it_e is not NULL:
            event_e = dbp_iterator_current(it_e)
            start = diff_time(reader_start, dbp_event_get_timestamp(event_s))
            end = diff_time(reader_start, dbp_event_get_timestamp(event_e))
            event = dbpEvent(thread, dbp_event_get_key(event_s), dbp_event_get_flags(event_s),
                             dbp_event_get_object_id(event_s), dbp_event_get_event_id(event_s),
                             start, end)
            thread.events.append(event)
            dbp_iterator_delete(it_e)
      dbp_iterator_next(it_s)
      event_s = dbp_iterator_current(it_s)

   dbp_iterator_delete(it_s)
   return thread

