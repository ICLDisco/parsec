/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "debug.h"

#if defined(USE_MPI)
# define DISTRIBUTED
#include <mpi.h>
typedef MPI_Datatype DAGuE_remote_dep_datatype_t;
extern DAGuE_remote_dep_datatype_t DAGuE_DEFAULT_DATA_TYPE;
#else
# undef DISTRIBUTED
typedef void DAGuE_remote_dep_datatype_t;
#endif

#include "assignment.h"
#include "lifo.h"
#include "execution_unit.h"
#include "datarepo.h"
#include "dague.h"

#define DAGuE_ACTION_INIT_REMOTE_DEPS    0x0100
#define DAGuE_ACTION_SEND_REMOTE_DEPS    0x0200
#define DAGuE_ACTION_RECV_REMOTE_DEPS    0x0400
#define DAGuE_ACTION_RELEASE_REMOTE_DEPS (DAGuE_ACTION_INIT_REMOTE_DEPS | DAGuE_ACTION_SEND_REMOTE_DEPS)
#define DAGuE_ACTION_GETDATA_REMOTE_DEPS (DAGuE_ACTION_INIT_REMOTE_DEPS | DAGuE_ACTION_RECV_REMOTE_DEPS)
#define DAGuE_ACTION_RELEASE_LOCAL_DEPS  0x0800
#define DAGuE_ACTION_GETTYPE_REMOTE_DEPS 0x1000
#define DAGuE_ACTION_NO_PLACEHOLDER      0x2000
#define DAGuE_ACTION_RELEASE_LOCAL_REFS  0x4000
#define DAGuE_ACTION_DEPS_MASK           0x00FF

typedef unsigned long remote_dep_datakey_t;

typedef struct remote_dep_wire_activate_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    remote_dep_datakey_t function;
    assignment_t locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    remote_dep_datakey_t tag;
} remote_dep_wire_get_t;

struct DAGuE_remote_deps_t {
    DAGuE_list_item_t                       item;
    struct DAGuE_atomic_lifo_t*             origin;
    remote_dep_wire_activate_t                msg;
    int                                       root;
    uint32_t                                  output_count;
    uint32_t                                  output_sent_count;
    struct { /** Never change this structure without understanding the 
              *   "subtle" relation with  remote_deps_allocation_init in remote_dep.c
              */
        gc_data_t*                            data;
        DAGuE_remote_dep_datatype_t*        type;
        uint32_t*                             rank_bits;
        uint32_t                              count;
    } output[1];
};


/* Gives pointers to expr_t allowing for evaluation of GRID predicates, needed 
 * by the precompiler only */
int DAGuE_remote_dep_get_rank_preds(const expr_t **predicates,
                                      expr_t **rowpred,
                                      expr_t **colpred, 
                                      symbol_t **rowsize,
                                      symbol_t **colsize);

#if defined(DISTRIBUTED)

extern DAGuE_atomic_lifo_t remote_deps_freelist;
extern uint32_t max_dep_count, max_nodes_number, elem_size;

int remote_deps_allocation_init(int np, int max_deps);

static inline DAGuE_remote_deps_t* remote_deps_allocation( DAGuE_atomic_lifo_t* lifo )
{
    uint32_t i, rank_bit_size;
    char *ptr;
    DAGuE_remote_deps_t* remote_deps = (DAGuE_remote_deps_t*)DAGuE_atomic_lifo_pop(lifo);
    if( NULL == remote_deps ) {
        remote_deps = (DAGuE_remote_deps_t*)calloc(1, elem_size);
        remote_deps->origin = lifo;
    }    
    ptr = (char*)(&(remote_deps->output[max_dep_count]));
    rank_bit_size = sizeof(uint32_t) * ((max_nodes_number + (8 * sizeof(uint32_t) - 1)) / (8*sizeof(uint32_t)));
    for( i = 0; i < max_dep_count; i++ ) {
        remote_deps->output[i].rank_bits = (uint32_t*)ptr;
        ptr += rank_bit_size;
    }
    assert( (ptr - (char*)remote_deps) <= elem_size );
    return remote_deps;
}
#define DAGuE_ALLOCATE_REMOTE_DEPS_IF_NULL(REMOTE_DEPS, EXEC_CONTEXT, COUNT) \
    if( NULL == (REMOTE_DEPS) ) { /* only once per function */                 \
        int _i;                                                                \
        (REMOTE_DEPS) = (DAGuE_remote_deps_t*)remote_deps_allocation(&remote_deps_freelist); \
        (REMOTE_DEPS)->origin = (DAGuE_atomic_lifo_t*)&remote_deps_freelist; \
    }


int DAGuE_remote_dep_init(DAGuE_context_t* context);
int DAGuE_remote_dep_fini(DAGuE_context_t* context);
int DAGuE_remote_dep_on(DAGuE_context_t* context);
int DAGuE_remote_dep_off(DAGuE_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int DAGuE_remote_dep_progress(DAGuE_execution_unit_t* eu_context);

/* Send remote dependencies to target processes */
int DAGuE_remote_dep_activate(DAGuE_execution_unit_t* eu_context,
                                const DAGuE_execution_context_t* origin,
                                DAGuE_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count );

/* Memcopy a particular data using datatype specification */
void DAGuE_remote_dep_memcpy(void *dst, gc_data_t *src, const DAGuE_remote_dep_datatype_t datatype);

/* Create a default datatype */
void remote_dep_mpi_create_default_datatype(int tile_size, DAGuE_remote_dep_datatype_t base);

#else 
# define DAGuE_remote_dep_init(ctx) (1)
# define DAGuE_remote_dep_fini(ctx) (0)
# define DAGuE_remote_dep_on(ctx)   (0)
# define DAGuE_remote_dep_off(ctx)  (0)
# define DAGuE_remote_dep_progress(ctx) (0)
# define DAGuE_remote_dep_activate(ctx, o, r, c) (-1)
# define DAGuE_DEFAULT_DATA_TYPE    (NULL)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

