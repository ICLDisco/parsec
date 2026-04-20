# PaRSEC C library declarations for Cython

cdef extern from "parsec.h":
    # Basic types
    ctypedef struct parsec_context_s:
        pass
    ctypedef parsec_context_s* parsec_context_t
    
    ctypedef struct parsec_taskpool_s:
        pass
    ctypedef parsec_taskpool_s* parsec_taskpool_t
    
    ctypedef struct parsec_task_s:
        pass
    ctypedef parsec_task_s* parsec_task_t
    
    ctypedef struct parsec_data_s:
        pass
    ctypedef parsec_data_s* parsec_data_t
    
    ctypedef struct parsec_data_copy_s:
        pass
    ctypedef parsec_data_copy_s* parsec_data_copy_t
    
    ctypedef struct parsec_data_collection_s:
        pass
    ctypedef parsec_data_collection_s* parsec_data_collection_t
    
    ctypedef struct parsec_execution_stream_s:
        pass
    ctypedef parsec_execution_stream_s* parsec_execution_stream_t
    
    ctypedef struct parsec_arena_s:
        pass
    ctypedef parsec_arena_s* parsec_arena_t
    
    ctypedef struct parsec_arena_datatype_s:
        pass
    ctypedef parsec_arena_datatype_s* parsec_arena_datatype_t
    
    # Data types
    ctypedef uint64_t parsec_data_key_t
    ctypedef uint8_t parsec_data_coherency_t
    ctypedef uint8_t parsec_data_status_t
    ctypedef uint8_t parsec_data_flag_t
    ctypedef int parsec_datatype_t
    
    # Context management
    parsec_context_t* parsec_init(int nb_cores, int* pargc, char** pargv[])
    int parsec_fini(parsec_context_t** pcontext)
    int parsec_context_start(parsec_context_t* context)
    int parsec_context_wait(parsec_context_t* context)
    int parsec_context_test(parsec_context_t* context)
    int parsec_context_add_taskpool(parsec_context_t* context, parsec_taskpool_t* tp)
    int parsec_context_remove_taskpool(parsec_taskpool_t* tp)
    int parsec_context_query(parsec_context_t* context, int cmd, ...)
    void parsec_abort(parsec_context_t* pcontext, int status)
    
    # Data management
    parsec_data_t* parsec_data_new()
    void parsec_data_delete(parsec_data_t* data)
    parsec_data_t* parsec_data_create(parsec_data_t** holder, parsec_data_collection_t* desc, 
                                     parsec_data_key_t key, void* ptr, size_t size, parsec_data_flag_t flags)
    void parsec_data_destroy(parsec_data_t* holder)
    void* parsec_data_get_ptr(parsec_data_t* data, uint32_t device)
    parsec_data_copy_t* parsec_data_get_copy(parsec_data_t* data, uint32_t device)
    parsec_data_copy_t* parsec_data_copy_new(parsec_data_t* data, uint8_t device, 
                                           parsec_datatype_t dtt, parsec_data_flag_t flags)
    void parsec_data_copy_release(parsec_data_copy_t* copy)
    void* parsec_data_copy_get_ptr(parsec_data_copy_t* data)
    
    # Taskpool management
    int parsec_taskpool_wait(parsec_taskpool_t* tp)
    int parsec_taskpool_test(parsec_taskpool_t* tp)
    void parsec_taskpool_free(parsec_taskpool_t* tp)
    int parsec_taskpool_reserve_id(parsec_taskpool_t* tp)
    int parsec_taskpool_register(parsec_taskpool_t* tp)
    void parsec_taskpool_unregister(parsec_taskpool_t* tp)
    parsec_taskpool_t* parsec_taskpool_lookup(uint32_t taskpool_id)
    
    # Execution stream
    parsec_execution_stream_t* parsec_my_execution_stream()
    
    # Version information
    int parsec_version(int* version_major, int* version_minor, int* version_release)
    int parsec_version_ex(size_t len, char* version_string)
    
    # Constants
    int PARSEC_SUCCESS
    int PARSEC_ERR_NOT_SUPPORTED
    int PARSEC_ERR_NOT_FOUND
    int PARSEC_ERR_VALUE_OUT_OF_BOUNDS
    
    # Data flags
    parsec_data_flag_t PARSEC_DATA_FLAG_ARENA
    parsec_data_flag_t PARSEC_DATA_FLAG_TRANSIT
    parsec_data_flag_t PARSEC_DATA_FLAG_EVICTED
    parsec_data_flag_t PARSEC_DATA_FLAG_PARSEC_MANAGED
    parsec_data_flag_t PARSEC_DATA_FLAG_PARSEC_OWNED
    
    # Data coherency
    parsec_data_coherency_t PARSEC_DATA_COHERENCY_INVALID
    parsec_data_coherency_t PARSEC_DATA_COHERENCY_OWNED
    parsec_data_coherency_t PARSEC_DATA_COHERENCY_EXCLUSIVE
    parsec_data_coherency_t PARSEC_DATA_COHERENCY_SHARED
    
    # Context query commands
    int PARSEC_CONTEXT_QUERY_NODES
    int PARSEC_CONTEXT_QUERY_RANK
    int PARSEC_CONTEXT_QUERY_DEVICES
    int PARSEC_CONTEXT_QUERY_DEVICES_FULL_PEER_ACCESS
    int PARSEC_CONTEXT_QUERY_CORES
    int PARSEC_CONTEXT_QUERY_ACTIVE_TASKPOOLS
