#include "hash_datadist.h"

#define DEFAULT_HASH_SIZE 65536

static inline uint32_t hash_hash(uint32_t hash_size, uint32_t key)
{
    return key % hash_size;
}

static uint32_t      hash_data_key(struct dague_ddesc_s *desc, ...);
static uint32_t      hash_rank_of(    dague_ddesc_t* ddesc, ... );
static uint32_t      hash_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t       hash_vpid_of(    dague_ddesc_t* ddesc, ... );
static int32_t       hash_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* hash_data_of(    dague_ddesc_t* ddesc, ... );
static dague_data_t* hash_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

dague_hash_datadist_t *dague_hash_datadist_create(int np, int myrank)
{
    dague_hash_datadist_t *o;

    o = (dague_hash_datadist_t*)malloc(sizeof(dague_hash_datadist_t));

    /* Super setup */
    o->super.nodes  = np;
    o->super.myrank = myrank;

    o->super.data_key      = hash_data_key;
    o->super.rank_of       = hash_rank_of;
    o->super.rank_of_key   = hash_rank_of_key;
    o->super.data_of       = hash_data_of;
    o->super.data_of_key   = hash_data_of_key;
    o->super.vpid_of       = hash_vpid_of;
    o->super.vpid_of_key   = hash_vpid_of_key;

#if defined(DAGUE_PROF_TRACE)
    o->super.key_to_string = hash_key_to_string;
    o->super.key_dim       = NULL;
    o->super.key           = NULL;
#endif

    o->hash_size = DEFAULT_HASH_SIZE;
    o->hash = (dague_hash_datadist_entry_t **)calloc(DEFAULT_HASH_SIZE,
                                                     sizeof(dague_hash_datadist_entry_t *));

    assert(vpmap_get_nb_vp() > 0);

    return o;
}

void dague_hash_datadist_destroy(dague_hash_datadist_t *d)
{
    int i;
    dague_hash_datadist_entry_t *n, *next;

    for(i = 0; i < d->hash_size; i++) {
        if( d->hash[i] != NULL ) {
            for(n = d->hash[i]; n!= NULL; n = next) {
                next = n->next;
                if( n->data != NULL ) {
                    OBJ_RELEASE(n->data);
                }
                    free(n);
            }
            d->hash[i] = NULL;
        }
    }
    free(d->hash);
    d->hash = NULL;
    d->hash_size = 0;
    dague_ddesc_destroy( &d->super );
    free(d);
}

static dague_hash_datadist_entry_t *hash_lookup(dague_hash_datadist_t *d, uint32_t key)
{
    dague_hash_datadist_entry_t *u;

    u = d->hash[ hash_hash(d->hash_size, key ) ];
    while(u != NULL) {
        if( u->key == key) {
            return u;
        }
        u = u->next;
    }
    return NULL;
}

static dague_hash_datadist_entry_t *hash_lookup_or_create(dague_hash_datadist_t *d, uint32_t key)
{
    dague_hash_datadist_entry_t *u = hash_lookup(d, key);
    uint32_t h;

    if( NULL != u ) {
        return u;
    }

    u = (dague_hash_datadist_entry_t*)malloc(sizeof(dague_hash_datadist_entry_t));
    memset(u, 0, sizeof(dague_hash_datadist_entry_t));
    u->key = key;

    h = hash_hash(d->hash_size, key);
    u->next = d->hash[h];
    d->hash[h] = u;

    return u;
}

void dague_hash_datadist_set_data(dague_hash_datadist_t *d, void *actual_data, uint32_t key, int vpid, int rank, uint32_t size)
{
    dague_hash_datadist_entry_t *u;

    u = hash_lookup_or_create(d, key);
    u->actual_data = actual_data;
    u->vpid = vpid;
    u->rank = rank;
    u->size = size;
}

static uint32_t      hash_data_key(struct dague_ddesc_s *desc, ...)
{
    uint32_t k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);
    return k;
}

static uint32_t      hash_rank_of(    dague_ddesc_t* ddesc, ... )
{
    uint32_t k;
    va_list ap;

    va_start(ap, ddesc);
    k = va_arg(ap, int);
    va_end(ap);
    return hash_rank_of_key(ddesc, k);
}

static uint32_t      hash_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
    dague_hash_datadist_entry_t *e = hash_lookup( (dague_hash_datadist_t*)ddesc, key );
    assert(e != NULL);
    return e->rank;
}

static int32_t       hash_vpid_of(    dague_ddesc_t* ddesc, ... )
{
    uint32_t k;
    va_list ap;

    va_start(ap, ddesc);
    k = va_arg(ap, int);
    va_end(ap);
    return hash_vpid_of_key(ddesc, k);
}

static int32_t       hash_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
    dague_hash_datadist_entry_t *e = hash_lookup( (dague_hash_datadist_t*)ddesc, key );
    assert(e != NULL);
    return e->vpid;
}

static dague_data_t* hash_data_of(    dague_ddesc_t* ddesc, ... )
{
    uint32_t k;
    va_list ap;

    va_start(ap, ddesc);
    k = va_arg(ap, int);
    va_end(ap);
    return hash_data_of_key(ddesc, k);
}

static dague_data_t* hash_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
    dague_hash_datadist_entry_t *e = hash_lookup( (dague_hash_datadist_t*)ddesc, key );
    dague_data_t* data;

    assert(e != NULL);
    data = e->data;

    if( data == NULL ) {
        dague_data_copy_t* data_copy = OBJ_NEW(dague_data_copy_t);
        data = OBJ_NEW(dague_data_t);

        data_copy->coherency_state = DATA_COHERENCY_OWNED;
        data_copy->original = data;
        data_copy->device_private = e->actual_data;

        data->owner_device = 0;
        data->key = key;
        data->nb_elts = e->size;
        data->device_copies[0] = data_copy;

        if( !dague_atomic_cas(&e->data, NULL, data) ) {
            free(data_copy);
            free(data);
            data = e->data;
        }
    }

    return data;
}
