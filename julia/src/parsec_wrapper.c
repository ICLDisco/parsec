#include <parsec.h>
#include <parsec/parsec_config.h>
#include <parsec/interfaces/dtd/insert_function.h>

// TILE_FULL should be a reference to the arena datatype, not a constant
// We'll pass it as a parameter to the wrapper function

// Wrapper function for parsec_dtd_unpack_args with 9 parameters (3 data + 6 value)
void parsec_dtd_unpack_args_gemm_wrapper(parsec_task_t* this_task, 
                                         void** tileA,
                                         void** tileB,
                                         void** tileC,
                                         int* m_val,
                                         int* n_val,
                                         int* k_val,
                                         int* mb_val,
                                         int* nb_val,
                                         int* kb_val) {
    printf("C wrapper: Unpacking 9 parameter task arguments (3 data + 6 value)\n");
    fflush(stdout);
    
    // Call the real parsec_dtd_unpack_args function with the variable arguments
    parsec_dtd_unpack_args(this_task,
                          PASSED_BY_REF, tileA, PARSEC_INPUT,
                          PASSED_BY_REF, tileB, PARSEC_INPUT,
                          PASSED_BY_REF, tileC, PARSEC_INOUT,
                          PARSEC_DTD_EMPTY_FLAG, m_val,
                          PARSEC_DTD_EMPTY_FLAG, n_val,
                          PARSEC_DTD_EMPTY_FLAG, k_val,
                          PARSEC_DTD_EMPTY_FLAG, mb_val,
                          PARSEC_DTD_EMPTY_FLAG, nb_val,
                          PARSEC_DTD_EMPTY_FLAG, kb_val,
                          PARSEC_DTD_ARG_END);
    
    printf("C wrapper: 9 parameter task arguments unpacked successfully\n");
    fflush(stdout);
}

// Wrapper function for parsec_dtd_create_task_class with fixed arguments for GEMM
parsec_task_class_t* parsec_dtd_create_task_class_gemm_wrapper(parsec_taskpool_t* tp, const char* name, parsec_arena_datatype_t* adt, int tile_full) {
    printf("C wrapper: Creating task class '%s' with 9 parameters\n", name);
    fflush(stdout);
    printf("C wrapper: Taskpool = %p, Arena datatype = %p\n", tp, adt);
    fflush(stdout);
    
    // Check if taskpool is valid
    if (tp == NULL) {
        printf("C wrapper: ERROR - taskpool is NULL\n");
        fflush(stdout);
        return NULL;
    }
    
    // Use the passed tile_full parameter
    printf("C wrapper: TILE_FULL = %d\n", tile_full);
    fflush(stdout);
    
    // Full GEMM implementation with 9 parameters (like the working example)
    printf("C wrapper: About to call parsec_dtd_create_task_class with 9 parameters\n");
    printf("C wrapper: PASSED_BY_REF=%d, PARSEC_INPUT=%d, TILE_FULL=%d\n", PASSED_BY_REF, PARSEC_INPUT, tile_full);
    printf("C wrapper: sizeof(int)=%zu, PARSEC_VALUE=%d\n", sizeof(int), PARSEC_VALUE);
    printf("C wrapper: PARSEC_DTD_ARG_END=%d\n", PARSEC_DTD_ARG_END);
    fflush(stdout);
    
    parsec_task_class_t* result = parsec_dtd_create_task_class(tp, name,
                                       PASSED_BY_REF, PARSEC_INPUT | tile_full, /* A  */
                                       PASSED_BY_REF, PARSEC_INPUT | tile_full, /* B  */
                                       PASSED_BY_REF, PARSEC_INOUT | tile_full | PARSEC_AFFINITY, /* C  */
                                       sizeof(int), PARSEC_VALUE,               /* m  */
                                       sizeof(int), PARSEC_VALUE,               /* n  */
                                       sizeof(int), PARSEC_VALUE,               /* k  */
                                       sizeof(int), PARSEC_VALUE,               /* mb */
                                       sizeof(int), PARSEC_VALUE,               /* nb */
                                       sizeof(int), PARSEC_VALUE,               /* kb */
                                       PARSEC_DTD_ARG_END);
    
    printf("C wrapper: parsec_dtd_create_task_class returned %p\n", result);
    fflush(stdout);
    
    if (result != NULL) {
        printf("C wrapper: Task class created successfully\n");
        // Try to get more info about the task class
        printf("C wrapper: Task class name: %s\n", name);
    } else {
        printf("C wrapper: Task class creation failed\n");
    }
    fflush(stdout);
    
    return result;
}

// Wrapper function for parsec_dtd_insert_task_with_task_class (correct approach)
void parsec_dtd_insert_task_gemm_wrapper(parsec_taskpool_t* tp, 
                                         parsec_task_class_t* tc,
                                         int priority,
                                         int device,
                                         const char* task_name,
                                         void* tileA,
                                         void* tileB,
                                         void* tileC,
                                         int* m_val,
                                         int* n_val,
                                         int* k_val,
                                         int* mb_val,
                                         int* nb_val,
                                         int* kb_val) {
    printf("C wrapper: Inserting task with task class using parsec_dtd_insert_task_with_task_class\n");
    fflush(stdout);
    printf("C wrapper: Taskpool = %p, Task class = %p\n", tp, tc);
    fflush(stdout);
    printf("C wrapper: Priority = %d, Device = %d, Task name = %s\n", priority, device, task_name);
    fflush(stdout);
    
    printf("C wrapper: About to call parsec_dtd_insert_task_with_task_class\n");
    printf("C wrapper: Insertion parameters:\n");
    printf("  Priority: %d\n", priority);
    printf("  Device: %d\n", device);
    printf("  Task name: %s\n", task_name);
    printf("  sizeof(int)=%zu, PARSEC_VALUE=%d\n", sizeof(int), PARSEC_VALUE);
    printf("  PARSEC_DTD_ARG_END=%d\n", PARSEC_DTD_ARG_END);
    fflush(stdout);
    
    parsec_dtd_insert_task_with_task_class(tp, tc, priority, device, task_name,
                                          PARSEC_INPUT, tileA,
                                          PARSEC_INPUT, tileB,
                                          PARSEC_INOUT, tileC,
                                          PARSEC_DTD_EMPTY_FLAG, m_val,
                                          PARSEC_DTD_EMPTY_FLAG, n_val,
                                          PARSEC_DTD_EMPTY_FLAG, k_val,
                                          PARSEC_DTD_EMPTY_FLAG, mb_val,
                                          PARSEC_DTD_EMPTY_FLAG, nb_val,
                                          PARSEC_DTD_EMPTY_FLAG, kb_val,
                                          PARSEC_DTD_ARG_END);
    
    printf("C wrapper: parsec_dtd_insert_task_with_task_class completed successfully\n");
    fflush(stdout);
}
