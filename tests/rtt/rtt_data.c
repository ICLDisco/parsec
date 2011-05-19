#include "rtt_data.h"
#include "stdarg.h"
#include "data_distribution.h"

typedef struct {
    dague_ddesc_t super;
    unsigned char *data;
} my_datatype_t;

static uint32_t rank_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    return k % dat->super.nodes;
}

static void *data_of(dague_ddesc_t *desc, ...)
{
    int k;

    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

    return (void*)(dat->data);
} 

dague_ddesc_t *create_and_distribute_data(int rank, int world, int cores, int size)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    dague_ddesc_t *d = &(m->super);

    d->myrank = rank;
    d->cores  = cores;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;

    m->data = (unsigned char *)malloc(size);

    return d;
}

void free_data(dague_ddesc_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    free(m->data);
    dague_ddesc_destroy(d);
    free(d);
}
