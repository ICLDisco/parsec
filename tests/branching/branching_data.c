#include "branching_data.h"
#include "stdarg.h"
#include "data_distribution.h"
#include "data.h"

typedef struct {
    dague_ddesc_t super;
    struct dague_data_copy_s* data;
    int32_t* ptr;
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

static int32_t vpid_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

    return 0;
}

static dague_data_t* data_of(dague_ddesc_t *desc, ...)
{
    int k;

    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

    if(NULL == dat->data) {
        dat->data = dague_data_copy_new(NULL, 0);
        dat->data->device_private = dat->ptr;
    }
    return (void*)(dat->data);
}

#if defined(DAGUE_PROF_TRACE)
static uint32_t data_key(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    return (uint32_t)k;
}
#endif

dague_ddesc_t *create_and_distribute_data(int rank, int world, int cores, int size)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    dague_ddesc_t *d = &(m->super);

    d->myrank = rank;
    d->cores  = cores;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(DAGUE_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", size);
    d->key_base = NULL;
    d->data_key = data_key;
#endif

    m->data = NULL;
    m->ptr = (int32_t*)malloc(size * sizeof(int32_t));

    return d;
}

void free_data(dague_ddesc_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    if(NULL != m->data) {
        DAGUE_DATA_COPY_RELEASE(m->data);
    }
    free(m->ptr);
    dague_ddesc_destroy(d);
    free(d);
}
