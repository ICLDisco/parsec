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
      self.thread_count = 0
      self.handle_counts = []

class dbpDictEntry:
   def __init__(self, id, attributes):
      self.id = id
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
   def __str__(self):
      return str(self.id)

class dbpEvent:
   __max_length__ = 0
   # keep the print order updated as attributes are added
   print_order = ['handle_id', 'thread', 'key', 'event_id', 
                  'flags', 'start', 'end', 'duration', 'info']
   def __init__(self, parentThread, key, flags, handle_id, event_id, start, end):
      self.handle_id = handle_id
      self.thread = parentThread
      self.key = key
      self.event_id = event_id
      self.flags = flags
      self.start = start
      self.end = end
      self.duration = self.end - self.start
      self.info = None
      for attr, value in vars(self).items():
         if len(attr) > dbpEvent.__max_length__:
            dbpEvent.__max_length__ = len(attr)
         # values that we don't want printed generically
         elif attr == 'info':
            value = 'Yes' if self.info else 'No'
         if len(str(value)) > dbpEvent.__max_length__:
            dbpEvent.__max_length__ = len(str(value))
      
   def row_header(self):
      # first, establish max length
      header = ''
      for attr in dbpEvent.print_order:
         header += ('{:>' + str(dbpEvent.__max_length__) + '}  ').format(attr)
      return header
   def __repr__(self):
      row = ''
      for attr in dbpEvent.print_order:
         value = vars(self)[attr]
         # values that we don't want printed generically
         if attr == 'info':
            value = 'Yes' if self.info else 'No'
         row += ('{:>' + str(dbpEvent.__max_length__) + '}  ').format(value)
      return row

# Cython-y code

# this is the public Python interface function. call it.
cpdef readProfile(filenames):
   cdef char ** c_filenames = stringListToCStrings(filenames)
   cdef dbp_multifile_reader_t * dbp = dbp_reader_open_files(len(filenames), c_filenames)
   reader = multifile_reader(dbp_reader_nb_files(dbp), dbp_reader_nb_dictionary_entries(dbp))
   cdef dbp_file_t * cfile
   cdef dbp_dictionary_t * cdict

   reader.worldsize = dbp_reader_worldsize(dbp)

   # create dictionary first, for later use while making Events
   for index in range(reader.nb_dict_entries):
      dico = dbp_reader_get_dictionary(dbp, index)
      entry = dbpDictEntry(index, dbp_dictionary_attributes(dico))
      reader.dictionary[dbp_dictionary_name(dico)] = entry

   # convert c to py
   for ifd in range(reader.nb_files):
      cfile = dbp_reader_get_file(dbp, ifd)
      file = dbpFile(reader, dbp_file_hr_id(cfile), dbp_file_get_name(cfile), dbp_file_get_rank(cfile))
      for index in range(dbp_file_nb_infos(cfile)):
         cinfo = dbp_file_get_info(cfile, index)
         key = dbp_info_get_key(cinfo)
         value = dbp_info_get_value(cinfo)
         file.infos.append(dbpInfo(key, value))
      for thread in range(dbp_file_nb_threads(cfile)):
         file.threads.append(makeDbpThread(reader, dbp, cfile, thread, file))
      reader.files.append(file)

   return reader

# helper function for readProfile
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

# you can't call this. it will be called for you. call readProfile()
cdef makeDbpThread(reader, dbp_multifile_reader_t * dbp, dbp_file_t * cfile, int index, file):
   cdef dbp_thread_t * cthread = dbp_file_get_thread(cfile, index)
   cdef dbp_event_iterator_t * it_s = dbp_iterator_new_from_thread(cthread)
   cdef dbp_event_iterator_t * it_e
   cdef dbp_event_t * event_s = dbp_iterator_current(it_s)
   cdef dbp_event_t * event_e
   cdef dague_time_t reader_start = dbp_reader_min_date(dbp)
   cdef unsigned long long start, end
   cdef void * cinfo
   cdef pins_cachemiss_info_t * cast_pins_info

   thread = dbpThread(file, index)
   if thread.id + 1 > reader.thread_count:
      reader.thread_count = thread.id + 1

   while event_s is not NULL:
      if KEY_IS_START( dbp_event_get_key(event_s) ):
         it_e = dbp_iterator_find_matching_event_all_threads(it_s)
         if it_e is not NULL:
            event_e = dbp_iterator_current(it_e)
            start = diff_time(reader_start, dbp_event_get_timestamp(event_s))
            end = diff_time(reader_start, dbp_event_get_timestamp(event_e))
            
            event = dbpEvent(thread,
                             dbp_event_get_key(event_s) / 2, # to match dictionary
                             dbp_event_get_flags(event_s),
                             dbp_event_get_handle_id(event_s), 
                             dbp_event_get_event_id(event_s),
                             start, end)

            while len(reader.handle_counts) < event.handle_id + 1:
               reader.handle_counts.append(0)
            reader.handle_counts[event.handle_id] += 1

            #####################################
            # not all events have info
            # also, not all events have the same info.
            # so this is where users must add code to translate
            # their own info objects
            cinfo = dbp_event_get_info(event_e)
            if cinfo != NULL:
               if event.key == reader.dictionary['PINS_EXEC_PAPI_CORE'].id:
                  cast_pins_info = <pins_cachemiss_info_t *>cinfo
                  event.info = dbp_ExecMisses_EventInfo(
	                  cast_pins_info.kernel_type, 
                          cast_pins_info.vp_id,
	                  cast_pins_info.th_id,
	                  [cast_pins_info.values[x] for x in range(cast_pins_info.values_len)])
               elif event.key == reader.dictionary['PINS_TASK_SELECT'].id:
                  cast_pins_info = <pins_task_select_info_t *>cinfo
                  event.info = dbp_TaskSelect_EventInfo(
                             cast_pins_info.kernel_type,
                             cast_pins_info.vp_id,
                             cast_pins_info.th_id,
                             cast_pins_info.victim_vp_id,
                             cast_pins_info.victim_th_id,
                             cast_pins_info.exec_context,
                             [cast_pins_info.values[x] for x in range(cast_pins_info.values_len)])
	       # elif event.key == reader.dictionary['<SOME OTHER TYPE WITH INFO>].id:
                  # event.info = <write a function and a Python type to translate>

            thread.events.append(event)
            dbp_iterator_delete(it_e)
      dbp_iterator_next(it_s)
      event_s = dbp_iterator_current(it_s)
   print('done processing events in thread {}'.format(thread.id))

   dbp_iterator_delete(it_s)
   return thread

########################################################
############## CUSTOM EVENT INFO SECTION ###############
######### -- add a Python type to this section #########
######### to allow for new 'info' types        #########

class dbp_ExecMisses_EventInfo:
   __max_length__ = 0
   def __init__(self, kernel_type, vp_id, th_id, values):
      self.kernel_type = kernel_type
      self.vp_id = vp_id
      self.th_id = th_id
      self.values = values

      # set global max length
      for attr, val in vars(self).items():
         if len(attr) > dbp_ExecMisses_EventInfo.__max_length__:
            dbp_ExecMisses_EventInfo.__max_length__ = len(attr)
         # values that we don't want printed generically
         elif attr == 'values':
            for value in val:
               if len(str(value)) > dbp_ExecMisses_EventInfo.__max_length__:
                  dbp_ExecMisses_EventInfo.__max_length__ = len(value)
         elif len(str(val)) > dbp_ExecMisses_EventInfo.__max_length__:
            dbp_ExecMisses_EventInfo.__max_length__ = len(str(val))

   def row_header(self):
      # first, establish max length
      header = ''
      length = str(dbp_ExecMisses_EventInfo.__max_length__)
      header += ('{:>' + length + '}  ').format('kernel_type')
      header += ('{:>' + length + '}  ').format('vp_id')
      header += ('{:>' + length + '}  ').format('th_id')
      header += ('{:>' + length + '}  ').format('values')
      return header

   def __repr__(self):
      rv = ''
      length = str(dbp_ExecMisses_EventInfo.__max_length__)
      rv += ('{:>' + length + '}  ').format(self.kernel_type)
      rv += ('{:>' + length + '}  ').format(self.vp_id)
      rv += ('{:>' + length + '}  ').format(self.th_id)
      for value in self.values:
         rv += ('{:>' + length + '}  ').format(value)
      return rv

class dbp_TaskSelect_EventInfo:
   __max_length__ = 0
   def __init__(self, kernel_type, vp_id, th_id, victim_vp_id, victim_th_id, exec_context, values):
      self.kernel_type = kernel_type
      self.vp_id = vp_id
      self.th_id = th_id
      self.victim_vp_id = victim_vp_id
      self.victim_th_id = victim_th_id
      self.exec_context = exec_context
      self.values = values

      # set global max length
      for attr, val in vars(self).items():
         if len(attr) > dbp_TaskSelect_EventInfo.__max_length__:
            dbp_TaskSelect_EventInfo.__max_length__ = len(attr)
         # values that we don't want printed generically
         elif attr == 'values':
            for value in val:
               if len(str(value)) > dbp_TaskSelect_EventInfo.__max_length__:
                  dbp_TaskSelect_EventInfo.__max_length__ = len(value)
         elif len(str(val)) > dbp_TaskSelect_EventInfo.__max_length__:
            dbp_TaskSelect_EventInfo.__max_length__ = len(str(val))

   def isStarvation(self):
      return self.exec_context == 0

   def isSystemQueueSteal(self):
      return self.victim_vp_id == SYSTEM_QUEUE_VP

   def row_header(self):
      # first, establish max length
      header = ''
      length = str(dbp_TaskSelect_EventInfo.__max_length__)
      header += ('{:>' + length + '}  ').format('kernel_type')
      header += ('{:>' + length + '}  ').format('vp_id')
      header += ('{:>' + length + '}  ').format('th_id')
      header += ('{:>' + length + '}  ').format('vict_vp_id')
      header += ('{:>' + length + '}  ').format('vict_th_id')
      header += ('{:>' + length + '}  ').format('exec_context')
      header += ('{:>' + length + '}  ').format('values')
      return header

   def __repr__(self):
      rv = ''
      length = str(dbp_TaskSelect_EventInfo.__max_length__)
      rv += ('{:>' + length + '}  ').format(self.kernel_type)
      rv += ('{:>' + length + '}  ').format(self.vp_id)
      rv += ('{:>' + length + '}  ').format(self.th_id)
      rv += ('{:>' + length + '}  ').format(self.victim_vp_id)
      rv += ('{:>' + length + '}  ').format(self.victim_th_id)
      rv += ('{:>' + length + '}  ').format(self.exec_context)

      for value in self.values:
         rv += ('{:>' + length + '}  ').format(value)
      return rv

