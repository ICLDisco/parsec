########################################################
############## CUSTOM EVENT INFO SECTION ###############
### --- add a function and/or a type to this section ###
#### to allow for new 'info' types                ######

cdef extern from "dague/mca/pins/papi_exec/pins_papi_exec.h":
   enum: NUM_EXEC_EVENTS # allows us to grab the #define from the .h
   enum: KERNEL_NAME_SIZE

   ctypedef struct papi_exec_info_t:
      int kernel_type
      char kernel_name[KERNEL_NAME_SIZE]
      int vp_id
      int th_id
      int values_len
      long long values[NUM_EXEC_EVENTS] # number is inconsequential

cdef extern from "dague/mca/pins/papi_select/pins_papi_select.h":
   enum: NUM_TASK_SELECT_EVENTS # allows us to grab the #define from the .h
   enum: SYSTEM_QUEUE_VP
   enum: KERNEL_NAME_SIZE

   ctypedef struct select_info_t:
      int kernel_type
      char kernel_name[KERNEL_NAME_SIZE]
      int vp_id
      int th_id
      int victim_vp_id
      int victim_th_id
      long long exec_context
      int values_len
      long long values[NUM_TASK_SELECT_EVENTS] # number is inconsequential

# cdef extern from "dague/mca/pins/papi_socket/pins_papi_socket.h":
#    enum: NUM_SOCKET_EVENTS # allows us to grab the #define from the .h

#    ctypedef struct papi_socket_info_t:
#       int vp_id
#       int th_id
#       int values_len
#       long long values[NUM_SOCKET_EVENTS] # number is inconsequential

cdef extern from "dague/mca/pins/papi_L123/pins_papi_L123.h":
   enum: SYSTEM_QUEUE_VP
   enum: NUM_CORE_EVENTS
   enum: NUM_SOCKET_EVENTS

   ctypedef struct papi_core_select_info_t:
      int kernel_type
      int victim_vp_id
      int victim_th_id
      long long selection_time
      long long exec_context
      long long evt_values[NUM_CORE_EVENTS]

   ctypedef struct papi_core_exec_info_t:
      int kernel_type
      long long evt_values[NUM_CORE_EVENTS]

   ctypedef struct papi_core_socket_info_t:
      long long evt_values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS]
