#ifndef gpu_data_h
#define gpu_data_h

/**
 * Enable GPU-compatible memory if possible
 */
void dague_data_enable_gpu( int nbgpu );

/**
 * returns not false iff dague_data_enable_gpu succeeded
 */
int dague_using_gpu(void);

/**
 * allocate a buffer to hold the data using GPU-compatible memory if needed
 */
void* dague_allocate_data( size_t matrix_size );

/**
 * free a buffer allocated by dague_allocate_data
 */
void dague_free_data(void *address);

#endif
