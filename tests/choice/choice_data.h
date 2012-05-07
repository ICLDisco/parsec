#ifndef choice_data_h
#define choice_data_h

#include <dague.h>
#include <data_distribution.h>

dague_ddesc_t *create_and_distribute_data(int rank, int world, int cores, int size);
void free_data(dague_ddesc_t *d);

#endif
