#include "dague_internal.h"
#include <dague.h>
#include <data_distribution.h>
#include <data_dist/matrix/matrix.h>
#include <data.h>
#include <dague/utils/mca_param.h>
#include <arena.h>

#define RED_OP(_A, _B) (_A + _B)
#define TYPE(_X) INT(_X)

#define INT(_X) ( (int *)(_X) )
#define OFFSET(_X,_i) ( *(TYPE(_X)+(_i)) )
#define REDUCE(_A, _B, _i) OFFSET(_B,_i) = RED_OP( OFFSET(_A,_i), OFFSET(_B,_i) )

int compute_offset(int N, int t)
{
    int i, cnt = 0, offset = 0;

    for (i = 0; i < 8 * sizeof(N); i++) {
	if ((1 << i) & N) {
	    cnt++;
	}
	if (cnt == t)
	    return offset;

	if ((1 << i) & N) {
	    offset += (1 << i);
	}
    }
    assert(0);
}

int count_bits(int N)
{
    int i, cnt = 0;
    for (i = 0; i < 8 * sizeof(N); i++) {
	if ((1 << i) & N) {
	    cnt++;
	}
    }
    return cnt;
}

int log_of_tree_size(int N, int t)
{
    int i, cnt = 0;
    for (i = 0; i < 8 * sizeof(N); i++) {
	if ((1 << i) & N) {
	    cnt++;
	}
	if (cnt == t)
	    return i;
    }
    assert(0);
}

int index_to_tree(int N, int idx)
{
    int i, cnt = 0;
    for (i = 0; i < 8 * sizeof(N); i++) {
	if ((1 << i) & N) {
	    cnt++;
	    if (idx < (1 << i))
		return cnt;
	    else
		idx -= (1 << i);
	}
    }
    assert(0);
}

int global_to_local_index(int N, int idx)
{
    int i;
    for (i = 0; i < 8 * sizeof(N); i++) {
	if ((1 << i) & N) {
	    if (idx < (1 << i))
		return idx;
	    else
		idx -= (1 << i);
	}
    }
    assert(0);
}


#include <dague.h>
#include "debug.h"
#include <scheduling.h>
#include <dague/mca/pins/pins.h>
#include <remote_dep.h>
#include <datarepo.h>
#include <data.h>
#include <dague_prof_grapher.h>
#include <mempool.h>
#include "BT_reduction.h"

#define DAGUE_BT_reduction_NB_FUNCTIONS 4
#define DAGUE_BT_reduction_NB_DATA 1
#if defined(DAGUE_PROF_GRAPHER)
#include "dague_prof_grapher.h"
#endif /* defined(DAGUE_PROF_GRAPHER) */
#include <mempool.h>
#include <alloca.h>
typedef struct __dague_BT_reduction_internal_handle {
    dague_BT_reduction_handle_t super;
    /* The ranges to compute the hash key */
    int LINE_TERMINATOR_j_range;
    int LINEAR_REDUC_i_range;
    int BT_REDUC_t_range;
    int BT_REDUC_s_range;
    int BT_REDUC_i_range;
    int REDUCTION_i_range;
    /* The list of data repositories */
    data_repo_t *LINE_TERMINATOR_repository;
    data_repo_t *LINEAR_REDUC_repository;
    data_repo_t *BT_REDUC_repository;
    data_repo_t *REDUCTION_repository;
} __dague_BT_reduction_internal_handle_t;

#if defined(DAGUE_PROF_TRACE)
static int BT_reduction_profiling_array[2 * DAGUE_BT_reduction_NB_FUNCTIONS] = { -1 };
#endif /* defined(DAGUE_PROF_TRACE) */
/* Globals */
#define NB (__dague_handle->super.NB)
#define NT (__dague_handle->super.NT)

/* Data Access Macros */
#define dataA(dataA0,dataA1)  (((dague_ddesc_t*)__dague_handle->super.dataA)->data_of((dague_ddesc_t*)__dague_handle->super.dataA, (dataA0), (dataA1)))


/* Functions Predicates */
#define LINE_TERMINATOR_pred(j, tree_count, i, offset) (((dague_ddesc_t*)(__dague_handle->super.dataA))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA, offset, 0))
#define LINEAR_REDUC_pred(tree_count, i, sz, offset) (((dague_ddesc_t*)(__dague_handle->super.dataA))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA, offset, 0))
#define BT_REDUC_pred(tree_count, t, sz, s, lvl, i, offset) (((dague_ddesc_t*)(__dague_handle->super.dataA))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA, (offset + (i * 2)), 0))
#define REDUCTION_pred(i, t, li, sz) (((dague_ddesc_t*)(__dague_handle->super.dataA))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA, i, 0))

/* Data Repositories */
#define LINE_TERMINATOR_repo (__dague_handle->LINE_TERMINATOR_repository)
#define LINEAR_REDUC_repo (__dague_handle->LINEAR_REDUC_repository)
#define BT_REDUC_repo (__dague_handle->BT_REDUC_repository)
#define REDUCTION_repo (__dague_handle->REDUCTION_repository)
/* Dependency Tracking Allocation Macro */
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \
do {                                                                                         \
  int _vmin = (vMIN);                                                                        \
  int _vmax = (vMAX);                                                                        \
  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  DEBUG3(("Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p\n",    \
           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP)));    \
  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \
  DAGUE_STAT_INCREASE(mem_bitarray,  sizeof(dague_dependencies_t) + STAT_MALLOC_OVERHEAD +   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  (DEPS)->symbol = (vSYMBOL);                                                                \
  (DEPS)->min = _vmin;                                                                       \
  (DEPS)->max = _vmax;                                                                       \
  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \
} while (0)

static inline int dague_imin(int a, int b)
{
    return (a <= b) ? a : b;
};

static inline int dague_imax(int a, int b)
{
    return (a >= b) ? a : b;
};

/* Release dependencies output macro */
#if DAGUE_DEBUG_VERBOSE != 0
#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    DEBUG(("thread %d VP %d explore deps from %s:%s to %s:%s (from rank %d to %d) base ptr %p\n",\
           (NULL != (EU) ? (EU)->th_id : -1), (NULL != (EU) ? (EU)->virtual_process->vp_id : -1),\
           DEPO, dague_snprintf_execution_context(tmp1, 128, (TASKO)),\
           DEPI, dague_snprintf_execution_context(tmp2, 128, (TASKI)), (RSRC), (RDST), (DATA)));\
  } while(0)
#define ACQUIRE_FLOW(TASKI, DEPI, FUNO, DEPO, LOCALS, PTR)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    DEBUG(("task %s acquires flow %s from %s %s data ptr %p\n",\
           dague_snprintf_execution_context(tmp1, 128, (TASKI)), (DEPI),\
           (DEPO), dague_snprintf_assignments(tmp2, 128, (FUNO), (LOCALS)), (PTR)));\
  } while(0)
#else
#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)
#define ACQUIRE_FLOW(TASKI, DEPI, TASKO, DEPO, LOCALS, PTR)
#endif
static inline int BT_reduction_inline_c_expr1_line_207(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task LINE_TERMINATOR */
    int j = assignments[0].value;
    int tree_count = assignments[1].value;
    int i = assignments[2].value;
    int offset = assignments[3].value;

    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;

    return compute_offset(NT, i);
}

static inline int BT_reduction_inline_c_expr2_line_205(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task LINE_TERMINATOR */
    int j = assignments[0].value;
    int tree_count = assignments[1].value;
    int i = assignments[2].value;
    int offset = assignments[3].value;

    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;

    return count_bits(NT);
}

static inline int BT_reduction_inline_c_expr3_line_158(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task LINEAR_REDUC */
    int tree_count = assignments[0].value;
    int i = assignments[1].value;
    int sz = assignments[2].value;
    int offset = assignments[3].value;

    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;

    return compute_offset(NT, i);
}

static inline int BT_reduction_inline_c_expr4_line_157(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task LINEAR_REDUC */
    int tree_count = assignments[0].value;
    int i = assignments[1].value;
    int sz = assignments[2].value;
    int offset = assignments[3].value;

    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;

    return log_of_tree_size(NT, i);
}

static inline int BT_reduction_inline_c_expr5_line_155(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task LINEAR_REDUC */
    int tree_count = assignments[0].value;
    int i = assignments[1].value;
    int sz = assignments[2].value;
    int offset = assignments[3].value;

    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;

    return count_bits(NT);
}

static inline int BT_reduction_inline_c_expr6_line_121(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task BT_REDUC */
    int tree_count = assignments[0].value;
    int t = assignments[1].value;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int lvl = assignments[4].value;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

    return compute_offset(NT, t);
}

static inline int BT_reduction_inline_c_expr7_line_120(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task BT_REDUC */
    int tree_count = assignments[0].value;
    int t = assignments[1].value;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int lvl = assignments[4].value;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

    return (1 << lvl) - 1;
}

static inline int BT_reduction_inline_c_expr8_line_117(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task BT_REDUC */
    int tree_count = assignments[0].value;
    int t = assignments[1].value;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int lvl = assignments[4].value;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

    return log_of_tree_size(NT, t);
}

static inline int BT_reduction_inline_c_expr9_line_115(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task BT_REDUC */
    int tree_count = assignments[0].value;
    int t = assignments[1].value;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int lvl = assignments[4].value;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

    return count_bits(NT);
}

static inline int BT_reduction_inline_c_expr10_line_97(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task REDUCTION */
    int i = assignments[0].value;
    int t = assignments[1].value;
    int li = assignments[2].value;
    int sz = assignments[3].value;

    (void) i;
    (void) t;
    (void) li;
    (void) sz;

    return log_of_tree_size(NT, t);
}

static inline int BT_reduction_inline_c_expr11_line_96(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task REDUCTION */
    int i = assignments[0].value;
    int t = assignments[1].value;
    int li = assignments[2].value;
    int sz = assignments[3].value;

    (void) i;
    (void) t;
    (void) li;
    (void) sz;

    return global_to_local_index(NT, i);
}

static inline int BT_reduction_inline_c_expr12_line_95(const dague_handle_t * __dague_handle_parent,
						       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    (void) __dague_handle;
    /* This inline C function was declared in the context of the task REDUCTION */
    int i = assignments[0].value;
    int t = assignments[1].value;
    int li = assignments[2].value;
    int sz = assignments[3].value;

    (void) i;
    (void) t;
    (void) li;
    (void) sz;

    return index_to_tree(NT, i);
}

static inline uint64_t LINE_TERMINATOR_hash(const __dague_BT_reduction_internal_handle_t * __dague_handle,
					    const assignment_t * assignments)
{
    uint64_t __h = 0;
    int j = assignments[0].value;
    int j_min = 0;
    int tree_count = assignments[1].value;
    (void) tree_count;
    int i = assignments[2].value;
    (void) i;
    int offset = assignments[3].value;
    (void) offset;
    __h += (j - j_min);
    (void) __dague_handle;
    return __h;
}

static inline uint64_t LINEAR_REDUC_hash(const __dague_BT_reduction_internal_handle_t * __dague_handle,
					 const assignment_t * assignments)
{
    uint64_t __h = 0;
    int tree_count = assignments[0].value;
    (void) tree_count;
    int i = assignments[1].value;
    int i_min = 1;
    int sz = assignments[2].value;
    (void) sz;
    int offset = assignments[3].value;
    (void) offset;
    __h += (i - i_min);
    (void) __dague_handle;
    return __h;
}

static inline uint64_t BT_REDUC_hash(const __dague_BT_reduction_internal_handle_t * __dague_handle,
				     const assignment_t * assignments)
{
    uint64_t __h = 0;
    int tree_count = assignments[0].value;
    (void) tree_count;
    int t = assignments[1].value;
    int t_min = 1;
    int sz = assignments[2].value;
    (void) sz;
    int s = assignments[3].value;
    int s_min = 1;
    int lvl = assignments[4].value;
    (void) lvl;
    int i = assignments[5].value;
    int i_min = 0;
    int offset = assignments[6].value;
    (void) offset;
    __h += (t - t_min);
    __h += (s - s_min) * __dague_handle->BT_REDUC_t_range;
    __h += (i - i_min) * __dague_handle->BT_REDUC_t_range * __dague_handle->BT_REDUC_s_range;
    (void) __dague_handle;
    return __h;
}

static inline uint64_t REDUCTION_hash(const __dague_BT_reduction_internal_handle_t * __dague_handle,
				      const assignment_t * assignments)
{
    uint64_t __h = 0;
    int i = assignments[0].value;
    int i_min = 0;
    int t = assignments[1].value;
    (void) t;
    int li = assignments[2].value;
    (void) li;
    int sz = assignments[3].value;
    (void) sz;
    __h += (i - i_min);
    (void) __dague_handle;
    return __h;
}

/** Predeclarations of the dague_function_t */
static const dague_function_t BT_reduction_LINE_TERMINATOR;
static const dague_function_t BT_reduction_LINEAR_REDUC;
static const dague_function_t BT_reduction_BT_REDUC;
static const dague_function_t BT_reduction_REDUCTION;
/** Predeclarations of the parameters */
static const dague_flow_t flow_of_BT_reduction_LINE_TERMINATOR_for_T;
static const dague_flow_t flow_of_BT_reduction_LINEAR_REDUC_for_C;
static const dague_flow_t flow_of_BT_reduction_LINEAR_REDUC_for_B;
static const dague_flow_t flow_of_BT_reduction_BT_REDUC_for_B;
static const dague_flow_t flow_of_BT_reduction_BT_REDUC_for_A;
static const dague_flow_t flow_of_BT_reduction_REDUCTION_for_A;
/******                                LINE_TERMINATOR                                ******/

static inline int minexpr_of_symb_BT_reduction_LINE_TERMINATOR_j_fct(const dague_handle_t * __dague_handle_parent,
								     const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t minexpr_of_symb_BT_reduction_LINE_TERMINATOR_j = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_LINE_TERMINATOR_j_fct}
};

static inline int maxexpr_of_symb_BT_reduction_LINE_TERMINATOR_j_fct(const dague_handle_t * __dague_handle_parent,
								     const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t maxexpr_of_symb_BT_reduction_LINE_TERMINATOR_j = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_LINE_TERMINATOR_j_fct}
};
static const symbol_t symb_BT_reduction_LINE_TERMINATOR_j = {.name = "j",.context_index = 0,.min =
	&minexpr_of_symb_BT_reduction_LINE_TERMINATOR_j,.max =
	&maxexpr_of_symb_BT_reduction_LINE_TERMINATOR_j,.cst_inc = 1,.expr_inc = NULL,.flags =
	DAGUE_SYMBOL_IS_STANDALONE };

static inline int expr_of_symb_BT_reduction_LINE_TERMINATOR_tree_count_fct(const dague_handle_t * __dague_handle_parent,
									   const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr2_line_205((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_LINE_TERMINATOR_tree_count = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINE_TERMINATOR_tree_count_fct}
};
static const symbol_t symb_BT_reduction_LINE_TERMINATOR_tree_count = {.name = "tree_count",.context_index = 1,.min =
	&expr_of_symb_BT_reduction_LINE_TERMINATOR_tree_count,.max =
	&expr_of_symb_BT_reduction_LINE_TERMINATOR_tree_count,.cst_inc = 0,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_LINE_TERMINATOR_i_fct(const dague_handle_t * __dague_handle_parent,
								  const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return tree_count;
}

static const expr_t expr_of_symb_BT_reduction_LINE_TERMINATOR_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINE_TERMINATOR_i_fct}
};
static const symbol_t symb_BT_reduction_LINE_TERMINATOR_i = {.name = "i",.context_index = 2,.min =
	&expr_of_symb_BT_reduction_LINE_TERMINATOR_i,.max = &expr_of_symb_BT_reduction_LINE_TERMINATOR_i,.cst_inc =
	0,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_LINE_TERMINATOR_offset_fct(const dague_handle_t * __dague_handle_parent,
								       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr1_line_207((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_LINE_TERMINATOR_offset = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINE_TERMINATOR_offset_fct}
};
static const symbol_t symb_BT_reduction_LINE_TERMINATOR_offset = {.name = "offset",.context_index = 3,.min =
	&expr_of_symb_BT_reduction_LINE_TERMINATOR_offset,.max =
	&expr_of_symb_BT_reduction_LINE_TERMINATOR_offset,.cst_inc = 0,.expr_inc = NULL,.flags = 0x0 };

static inline int affinity_of_BT_reduction_LINE_TERMINATOR(dague_execution_context_t * this_task,
							   dague_data_ref_t * ref)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    int j = this_task->locals[0].value;
    int tree_count = this_task->locals[1].value;
    int i = this_task->locals[2].value;
    int offset = this_task->locals[3].value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void) __dague_handle;
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;
    ref->ddesc = (dague_ddesc_t *) __dague_handle->super.dataA;
    /* Compute data key */
    ref->key = ref->ddesc->data_key(ref->ddesc, offset, 0);
    return 1;
}

static inline int expr_of_p1_for_flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211_fct(const dague_handle_t *
												__dague_handle_parent,
												const assignment_t *
												assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return tree_count;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211_fct}
};

static const dep_t flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211 = {
    .cond = NULL,
    .ctl_gather_nb = NULL,
    .function_id = 2,		/* BT_reduction_LINEAR_REDUC */
    .flow = &flow_of_BT_reduction_LINEAR_REDUC_for_B,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINE_TERMINATOR_for_T,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211}
};

static const dague_flow_t flow_of_BT_reduction_LINE_TERMINATOR_for_T = {
    .name = "T",
    .sym_type = SYM_OUT,
    .flow_flags = FLOW_ACCESS_WRITE,
    .flow_index = 0,
    .flow_datatype_mask = 0x1,
    .dep_in = {NULL},
    .dep_out = {&flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211}
};

static void
iterate_successors_of_BT_reduction_LINE_TERMINATOR(dague_execution_unit_t * eu,
						   const dague_execution_context_t * this_task, uint32_t action_mask,
						   dague_ontask_function_t * ontask, void *ontask_arg)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    dague_execution_context_t nc;
    dague_dep_data_description_t data;
    int vpid_dst = -1, rank_src = 0, rank_dst = 0;
    int j = this_task->locals[0].value;
    int tree_count = this_task->locals[1].value;
    int i = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) rank_src;
    (void) rank_dst;
    (void) __dague_handle;
    (void) vpid_dst;
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;
    nc.dague_handle = this_task->dague_handle;
    nc.priority = this_task->priority;
    nc.chore_id = 0;
#if defined(DISTRIBUTED)
    rank_src =
	((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.dataA, offset,
								 0);
#endif
    if (action_mask & 0x1) {	/* Flow of Data T */
	data.data = this_task->data[0].data_out;
	data.arena = __dague_handle->super.arenas[DAGUE_BT_reduction_DEFAULT_ARENA];
	data.layout = data.arena->opaque_dtt;
	data.count = 1;
	data.displ = 0;
	nc.function = __dague_handle->super.super.functions_array[BT_reduction_LINEAR_REDUC.function_id];
	const int LINEAR_REDUC_tree_count =
	    BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, nc.locals);
	nc.locals[0].value = LINEAR_REDUC_tree_count;
	const int LINEAR_REDUC_i = tree_count;
	if ((LINEAR_REDUC_i >= (1)) && (LINEAR_REDUC_i <= (LINEAR_REDUC_tree_count))) {
	    nc.locals[1].value = LINEAR_REDUC_i;
	    const int LINEAR_REDUC_sz =
		BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, nc.locals);
	    nc.locals[2].value = LINEAR_REDUC_sz;
	    const int LINEAR_REDUC_offset =
		BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, nc.locals);
	    nc.locals[3].value = LINEAR_REDUC_offset;
#if defined(DISTRIBUTED)
	    rank_dst =
		((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
									 LINEAR_REDUC_offset, 0);
	    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
		vpid_dst =
		    ((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->super.
									     dataA, LINEAR_REDUC_offset, 0);
	    nc.priority = __dague_handle->super.super.priority;
	    RELEASE_DEP_OUTPUT(eu, "T", this_task, "B", &nc, rank_src, rank_dst, &data);
	    if (DAGUE_ITERATE_STOP ==
		ontask(eu, &nc, this_task, &flow_of_BT_reduction_LINE_TERMINATOR_for_T_dep1_atline_211, &data, rank_src,
		       rank_dst, vpid_dst, ontask_arg))
		return;
	}
    }
    (void) data;
    (void) nc;
    (void) eu;
    (void) ontask;
    (void) ontask_arg;
    (void) rank_dst;
    (void) action_mask;
}

static int release_deps_of_BT_reduction_LINE_TERMINATOR(dague_execution_unit_t * eu,
							dague_execution_context_t * context, uint32_t action_mask,
							dague_remote_deps_t * deps)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) context->dague_handle;
    dague_release_dep_fct_arg_t arg;
    int __vp_id;
    arg.action_mask = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
#if defined(DISTRIBUTED)
    arg.remote_deps = deps;
#endif /* defined(DISTRIBUTED) */
    assert(NULL != eu);
    arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);
    (void) __dague_handle;
    (void) deps;
    if (action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY)) {
	arg.output_entry =
	    data_repo_lookup_entry_and_create(eu, LINE_TERMINATOR_repo,
					      LINE_TERMINATOR_hash(__dague_handle, context->locals));
	arg.output_entry->generator = (void *) context;	/* for AYU */
#if defined(DAGUE_SIM)
	assert(arg.output_entry->sim_exec_date == 0);
	arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
    }
    iterate_successors_of_BT_reduction_LINE_TERMINATOR(eu, context, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
    if ((action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
	dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps->outgoing_mask);
    }
#endif

    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
	struct dague_vp_s **vps = eu->virtual_process->dague_context->virtual_processes;
	data_repo_entry_addto_usage_limit(LINE_TERMINATOR_repo, arg.output_entry->key, arg.output_usage);
	for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
	    if (NULL == arg.ready_lists[__vp_id])
		continue;
	    if (__vp_id == eu->virtual_process->vp_id) {
		__dague_schedule(eu, arg.ready_lists[__vp_id]);
	    } else {
		__dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
	    }
	    arg.ready_lists[__vp_id] = NULL;
	}
    }
    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
	if (NULL != context->data[0].data_in)
	    DAGUE_DATA_COPY_RELEASE(context->data[0].data_in);
    }
    return 0;
}

static int data_lookup_of_BT_reduction_LINE_TERMINATOR(dague_execution_unit_t * context,
						       dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    int target_device = 0;
    (void) target_device;
    (void) __dague_handle;
    (void) tass;
    (void) context;
    dague_data_copy_t *chunk = NULL;
    data_repo_entry_t *entry = NULL;
    int j = this_task->locals[0].value;
    int tree_count = this_task->locals[1].value;
    int i = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;
    (void) chunk;
    (void) entry;

  /** Lookup the input data, and store them in the context if any */
    if (NULL == (chunk = this_task->data[0].data_in)) {	/* flow T */
	entry = NULL;
	chunk = dague_arena_get_copy(__dague_handle->super.arenas[DAGUE_BT_reduction_DEFAULT_ARENA], 1, target_device);
	this_task->data[0].data_in = chunk;	/* flow T */
	this_task->data[0].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if (NULL != chunk)
	this_task->data[0].data_out = dague_data_get_copy(chunk->original, target_device);
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
    this_task->prof_info.desc = (dague_ddesc_t *) __dague_handle->super.dataA;
    this_task->prof_info.id =
	((dague_ddesc_t *) (__dague_handle->super.dataA))->data_key((dague_ddesc_t *) __dague_handle->super.dataA,
								    offset, 0);
#endif /* defined(DAGUE_PROF_TRACE) */
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;
    (void) chunk;
    (void) entry;

    return DAGUE_HOOK_RETURN_DONE;
}

static int hook_of_BT_reduction_LINE_TERMINATOR(dague_execution_unit_t * context, dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    (void) context;
    (void) __dague_handle;
    (void) tass;
    int j = this_task->locals[0].value;
    int tree_count = this_task->locals[1].value;
    int i = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;

  /** Declare the variables that will hold the data, and all the accounting for each */
    dague_data_copy_t *gT = this_task->data[0].data_in;
    void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL;
    (void) T;

  /** Update staring simulation date */
#if defined(DAGUE_SIM)
    this_task->sim_exec_date = 0;
    if ((NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eT->sim_exec_date;
    if (this_task->function->sim_cost_fct != NULL) {
	this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if (context->largest_simulation_date < this_task->sim_exec_date)
	context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
    cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              LINE_TERMINATOR BODY                              -----*/

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id], this_task);
    /* nothing */


/*-----                          END OF LINE_TERMINATOR BODY                          -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY) */

    return DAGUE_HOOK_RETURN_DONE;
}

static int complete_hook_of_BT_reduction_LINE_TERMINATOR(dague_execution_unit_t * context,
							 dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
#if defined(DISTRIBUTED)
    int j = this_task->locals[0].value;
    int tree_count = this_task->locals[1].value;
    int i = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
#endif /* defined(DISTRIBUTED) */
    (void) context;
    (void) __dague_handle;
    if (NULL != this_task->data[0].data_out)
	this_task->data[0].data_out->version++;
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
			  this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
			    LINE_TERMINATOR_hash(__dague_handle, this_task->locals));
#endif /* defined(DAGUE_PROF_GRAPHER) */
    release_deps_of_BT_reduction_LINE_TERMINATOR(context, this_task, DAGUE_ACTION_RELEASE_REMOTE_DEPS | DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_LOCAL_REFS | 0x1,	/* mask of all dep_index */
						 NULL);
    return 0;
}

static int BT_reduction_LINE_TERMINATOR_internal_init(__dague_BT_reduction_internal_handle_t * __dague_handle)
{
    dague_dependencies_t *dep = NULL;
    assignment_t assignments[MAX_LOCAL_COUNT];
    (void) assignments;
    int nb_tasks = 0;
    int32_t j, tree_count, i, offset;
    int32_t j_min = 0x7fffffff;
    int32_t j_max = 0;
    (void) __dague_handle;
    /* First, find the min and max value for each of the dimensions */
    for (j = 0; j <= 0; j += 1) {
	assignments[0].value = j;
	tree_count = BT_reduction_inline_c_expr2_line_205((const dague_handle_t *) __dague_handle, assignments);
	assignments[1].value = tree_count;
	i = tree_count;
	assignments[2].value = i;
	offset = BT_reduction_inline_c_expr1_line_207((const dague_handle_t *) __dague_handle, assignments);
	assignments[3].value = offset;
	j_max = dague_imax(j_max, j);
	j_min = dague_imin(j_min, j);
	if (!LINE_TERMINATOR_pred(j, tree_count, i, offset))
	    continue;
	nb_tasks++;
    }

  /**
   * Set the range variables for the collision-free hash-computation
   */
    __dague_handle->LINE_TERMINATOR_j_range = (j_max - j_min) + 1;

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    DEBUG3(("Allocating dependencies array for BT_reduction_LINE_TERMINATOR_internal_init\n"));
    if (0 != nb_tasks) {
	ALLOCATE_DEP_TRACKING(dep, j_min, j_max, "j", &symb_BT_reduction_LINE_TERMINATOR_j, NULL,
			      DAGUE_DEPENDENCIES_FLAG_FINAL);
    }

    AYU_REGISTER_TASK(&BT_reduction_LINE_TERMINATOR);
    __dague_handle->super.super.dependencies_array[3] = dep;
    __dague_handle->super.super.nb_local_tasks += nb_tasks;
    return nb_tasks;
}

static int BT_reduction_LINE_TERMINATOR_startup_tasks(dague_context_t * context,
						      __dague_BT_reduction_internal_handle_t * __dague_handle,
						      dague_execution_context_t ** pready_list)
{
    dague_execution_context_t *new_context, new_context_holder, *new_dynamic_context;
    assignment_t *assignments = NULL;
    int vpid = 0;
    int32_t j = -1, tree_count = -1, i = -1, offset = -1;
    (void) j;
    (void) tree_count;
    (void) i;
    (void) offset;
    new_context = &new_context_holder;
    assignments = new_context->locals;
    /* Parse all the inputs and generate the ready execution tasks */
    new_context->dague_handle = (dague_handle_t *) __dague_handle;
    new_context->function = __dague_handle->super.super.functions_array[BT_reduction_LINE_TERMINATOR.function_id];
    for (j = 0; j <= 0; j += 1) {
	assignments[0].value = j;
	assignments[1].value = tree_count =
	    BT_reduction_inline_c_expr2_line_205((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = i = tree_count;
	assignments[3].value = offset =
	    BT_reduction_inline_c_expr1_line_207((const dague_handle_t *) __dague_handle, assignments);
	if (!LINE_TERMINATOR_pred(j, tree_count, i, offset))
	    continue;
	if (NULL != ((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of) {
	    vpid =
		((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
									 offset, 0);
	    assert(context->nb_vp >= vpid);
	}
	new_dynamic_context =
	    (dague_execution_context_t *) dague_lifo_pop(&context->virtual_processes[vpid]->execution_units[0]->
							 context_mempool->mempool);
	if (NULL == new_dynamic_context)
	    new_dynamic_context =
		(dague_execution_context_t *) dague_thread_mempool_allocate(context->virtual_processes[0]->
									    execution_units[0]->context_mempool);
	/* Copy only the valid elements from new_context to new_dynamic one */
	new_dynamic_context->dague_handle = new_context->dague_handle;
	new_dynamic_context->function = new_context->function;
	new_dynamic_context->chore_id = 0;
	memcpy(new_dynamic_context->locals, new_context->locals, 4 * sizeof(assignment_t));
	DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
	DAGUE_LIST_ITEM_SINGLETON(new_dynamic_context);
	new_dynamic_context->priority = __dague_handle->super.super.priority;
	new_dynamic_context->data[0].data_repo = NULL;
	new_dynamic_context->data[0].data_in = NULL;
	new_dynamic_context->data[0].data_out = NULL;
#if DAGUE_DEBUG_VERBOSE != 0
	{
	    char tmp[128];
	    DEBUG2(("Add startup task %s\n", dague_snprintf_execution_context(tmp, 128, new_dynamic_context)));
	}
#endif
	dague_dependencies_mark_task_as_startup(new_dynamic_context);
	if (NULL != pready_list[vpid]) {
	    dague_list_item_ring_merge((dague_list_item_t *) new_dynamic_context,
				       (dague_list_item_t *) (pready_list[vpid]));
	}
	pready_list[vpid] = new_dynamic_context;
    }
    return 0;
}

static const __dague_chore_t __BT_reduction_LINE_TERMINATOR_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = hook_of_BT_reduction_LINE_TERMINATOR},
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = NULL},		/* End marker */
};

static const dague_function_t BT_reduction_LINE_TERMINATOR = {
    .name = "LINE_TERMINATOR",
    .function_id = 3,
    .nb_flows = 1,
    .nb_parameters = 1,
    .nb_locals = 4,
    .params = {&symb_BT_reduction_LINE_TERMINATOR_j, NULL},
    .locals =
	{&symb_BT_reduction_LINE_TERMINATOR_j, &symb_BT_reduction_LINE_TERMINATOR_tree_count,
	 &symb_BT_reduction_LINE_TERMINATOR_i, &symb_BT_reduction_LINE_TERMINATOR_offset, NULL},
    .data_affinity = affinity_of_BT_reduction_LINE_TERMINATOR,
    .initial_data = NULL,
    .final_data = NULL,
    .priority = NULL,
    .in = {NULL},
    .out = {&flow_of_BT_reduction_LINE_TERMINATOR_for_T, NULL},
    .flags = 0x0 | DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x0,
    .init = (dague_create_function_t *) NULL,
    .key = (dague_functionkey_fn_t *) LINE_TERMINATOR_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __BT_reduction_LINE_TERMINATOR_chores,
    .iterate_successors = iterate_successors_of_BT_reduction_LINE_TERMINATOR,
    .release_deps = release_deps_of_BT_reduction_LINE_TERMINATOR,
    .prepare_input = data_lookup_of_BT_reduction_LINE_TERMINATOR,
    .prepare_output = NULL,
    .complete_execution = complete_hook_of_BT_reduction_LINE_TERMINATOR,
#if defined(DAGUE_SIM)
    .sim_cost_fct = NULL,
#endif
};


/******                                  LINEAR_REDUC                                  ******/

static inline int expr_of_symb_BT_reduction_LINEAR_REDUC_tree_count_fct(const dague_handle_t * __dague_handle_parent,
									const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_LINEAR_REDUC_tree_count = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINEAR_REDUC_tree_count_fct}
};
static const symbol_t symb_BT_reduction_LINEAR_REDUC_tree_count = {.name = "tree_count",.context_index = 0,.min =
	&expr_of_symb_BT_reduction_LINEAR_REDUC_tree_count,.max =
	&expr_of_symb_BT_reduction_LINEAR_REDUC_tree_count,.cst_inc = 0,.expr_inc = NULL,.flags = 0x0 };

static inline int minexpr_of_symb_BT_reduction_LINEAR_REDUC_i_fct(const dague_handle_t * __dague_handle_parent,
								  const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t minexpr_of_symb_BT_reduction_LINEAR_REDUC_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_LINEAR_REDUC_i_fct}
};

static inline int maxexpr_of_symb_BT_reduction_LINEAR_REDUC_i_fct(const dague_handle_t * __dague_handle_parent,
								  const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[0].value;

    (void) __dague_handle;
    (void) assignments;
    return tree_count;
}

static const expr_t maxexpr_of_symb_BT_reduction_LINEAR_REDUC_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_LINEAR_REDUC_i_fct}
};
static const symbol_t symb_BT_reduction_LINEAR_REDUC_i = {.name = "i",.context_index = 1,.min =
	&minexpr_of_symb_BT_reduction_LINEAR_REDUC_i,.max = &maxexpr_of_symb_BT_reduction_LINEAR_REDUC_i,.cst_inc =
	1,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_LINEAR_REDUC_sz_fct(const dague_handle_t * __dague_handle_parent,
								const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_LINEAR_REDUC_sz = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINEAR_REDUC_sz_fct}
};
static const symbol_t symb_BT_reduction_LINEAR_REDUC_sz = {.name = "sz",.context_index = 2,.min =
	&expr_of_symb_BT_reduction_LINEAR_REDUC_sz,.max = &expr_of_symb_BT_reduction_LINEAR_REDUC_sz,.cst_inc =
	0,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_LINEAR_REDUC_offset_fct(const dague_handle_t * __dague_handle_parent,
								    const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_LINEAR_REDUC_offset = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_LINEAR_REDUC_offset_fct}
};
static const symbol_t symb_BT_reduction_LINEAR_REDUC_offset = {.name = "offset",.context_index = 3,.min =
	&expr_of_symb_BT_reduction_LINEAR_REDUC_offset,.max = &expr_of_symb_BT_reduction_LINEAR_REDUC_offset,.cst_inc =
	0,.expr_inc = NULL,.flags = 0x0 };

static inline int affinity_of_BT_reduction_LINEAR_REDUC(dague_execution_context_t * this_task, dague_data_ref_t * ref)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    int tree_count = this_task->locals[0].value;
    int i = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int offset = this_task->locals[3].value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void) __dague_handle;
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;
    ref->ddesc = (dague_ddesc_t *) __dague_handle->super.dataA;
    /* Compute data key */
    ref->key = ref->ddesc->data_key(ref->ddesc, offset, 0);
    return 1;
}

static inline int expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164_fct(const dague_handle_t *
											       __dague_handle_parent,
											       const assignment_t *
											       assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return (0 == sz);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int offset = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return offset;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164_fct}
};

static const dep_t flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164,
    .ctl_gather_nb = NULL,
    .function_id = 0,		/* BT_reduction_REDUCTION */
    .flow = &flow_of_BT_reduction_REDUCTION_for_A,
    .dep_index = 2,
    .dep_datatype_index = 2,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct(const dague_handle_t *
											       __dague_handle_parent,
											       const assignment_t *
											       assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return (0 < sz);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return i;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return sz;
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165_fct}
};

static const dep_t flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_B,
    .dep_index = 3,
    .dep_datatype_index = 2,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165,
		    &expr_of_p2_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165,
		    &expr_of_p3_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166_fct(const dague_handle_t *
											       __dague_handle_parent,
											       const assignment_t *
											       assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return (1 < i);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return (i - 1);
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166_fct}
};

static const dep_t flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166,
    .ctl_gather_nb = NULL,
    .function_id = 2,		/* BT_reduction_LINEAR_REDUC */
    .flow = &flow_of_BT_reduction_LINEAR_REDUC_for_B,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166}
};

static const dague_flow_t flow_of_BT_reduction_LINEAR_REDUC_for_C = {
    .name = "C",
    .sym_type = SYM_INOUT,
    .flow_flags = FLOW_ACCESS_RW,
    .flow_index = 0,
    .flow_datatype_mask = 0x1,
    .dep_in = {&flow_of_BT_reduction_LINEAR_REDUC_for_C_dep1_atline_164,
	       &flow_of_BT_reduction_LINEAR_REDUC_for_C_dep2_atline_165},
    .dep_out = {&flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162_fct(const dague_handle_t *
											       __dague_handle_parent,
											       const assignment_t *
											       assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[0].value;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return (tree_count == i);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162_fct}
};

static const dep_t flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162,
    .ctl_gather_nb = NULL,
    .function_id = 3,		/* BT_reduction_LINE_TERMINATOR */
    .flow = &flow_of_BT_reduction_LINE_TERMINATOR_for_T,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINEAR_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163_fct(const dague_handle_t *
											       __dague_handle_parent,
											       const assignment_t *
											       assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[0].value;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return (tree_count < i);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163_fct(const dague_handle_t *
											     __dague_handle_parent,
											     const assignment_t *
											     assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return (i + 1);
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163_fct}
};

static const dep_t flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163,
    .ctl_gather_nb = NULL,
    .function_id = 2,		/* BT_reduction_LINEAR_REDUC */
    .flow = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .dep_index = 1,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_LINEAR_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163}
};

static const dague_flow_t flow_of_BT_reduction_LINEAR_REDUC_for_B = {
    .name = "B",
    .sym_type = SYM_IN,
    .flow_flags = FLOW_ACCESS_READ,
    .flow_index = 1,
    .flow_datatype_mask = 0x0,
    .dep_in = {&flow_of_BT_reduction_LINEAR_REDUC_for_B_dep1_atline_162,
	       &flow_of_BT_reduction_LINEAR_REDUC_for_B_dep2_atline_163},
    .dep_out = {NULL}
};

static void
iterate_successors_of_BT_reduction_LINEAR_REDUC(dague_execution_unit_t * eu,
						const dague_execution_context_t * this_task, uint32_t action_mask,
						dague_ontask_function_t * ontask, void *ontask_arg)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    dague_execution_context_t nc;
    dague_dep_data_description_t data;
    int vpid_dst = -1, rank_src = 0, rank_dst = 0;
    int tree_count = this_task->locals[0].value;
    int i = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) rank_src;
    (void) rank_dst;
    (void) __dague_handle;
    (void) vpid_dst;
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;
    nc.dague_handle = this_task->dague_handle;
    nc.priority = this_task->priority;
    nc.chore_id = 0;
#if defined(DISTRIBUTED)
    rank_src =
	((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.dataA, offset,
								 0);
#endif
    if (action_mask & 0x1) {	/* Flow of Data C */
	data.data = this_task->data[0].data_out;
	data.arena = __dague_handle->super.arenas[DAGUE_BT_reduction_DEFAULT_ARENA];
	data.layout = data.arena->opaque_dtt;
	data.count = 1;
	data.displ = 0;
	if ((1 < i)) {
	    nc.function = __dague_handle->super.super.functions_array[BT_reduction_LINEAR_REDUC.function_id];
	    const int LINEAR_REDUC_tree_count =
		BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, nc.locals);
	    nc.locals[0].value = LINEAR_REDUC_tree_count;
	    const int LINEAR_REDUC_i = (i - 1);
	    if ((LINEAR_REDUC_i >= (1)) && (LINEAR_REDUC_i <= (LINEAR_REDUC_tree_count))) {
		nc.locals[1].value = LINEAR_REDUC_i;
		const int LINEAR_REDUC_sz =
		    BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[2].value = LINEAR_REDUC_sz;
		const int LINEAR_REDUC_offset =
		    BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[3].value = LINEAR_REDUC_offset;
#if defined(DISTRIBUTED)
		rank_dst =
		    ((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.
									     dataA, LINEAR_REDUC_offset, 0);
		if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
		    vpid_dst =
			((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->
										 super.dataA, LINEAR_REDUC_offset, 0);
		nc.priority = __dague_handle->super.super.priority;
		RELEASE_DEP_OUTPUT(eu, "C", this_task, "B", &nc, rank_src, rank_dst, &data);
		if (DAGUE_ITERATE_STOP ==
		    ontask(eu, &nc, this_task, &flow_of_BT_reduction_LINEAR_REDUC_for_C_dep3_atline_166, &data,
			   rank_src, rank_dst, vpid_dst, ontask_arg))
		    return;
	    }
	}
    }
    /* Flow of data B has only IN dependencies */
    (void) data;
    (void) nc;
    (void) eu;
    (void) ontask;
    (void) ontask_arg;
    (void) rank_dst;
    (void) action_mask;
}

static int release_deps_of_BT_reduction_LINEAR_REDUC(dague_execution_unit_t * eu, dague_execution_context_t * context,
						     uint32_t action_mask, dague_remote_deps_t * deps)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) context->dague_handle;
    dague_release_dep_fct_arg_t arg;
    int __vp_id;
    arg.action_mask = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
#if defined(DISTRIBUTED)
    arg.remote_deps = deps;
#endif /* defined(DISTRIBUTED) */
    assert(NULL != eu);
    arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);
    (void) __dague_handle;
    (void) deps;
    if (action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY)) {
	arg.output_entry =
	    data_repo_lookup_entry_and_create(eu, LINEAR_REDUC_repo,
					      LINEAR_REDUC_hash(__dague_handle, context->locals));
	arg.output_entry->generator = (void *) context;	/* for AYU */
#if defined(DAGUE_SIM)
	assert(arg.output_entry->sim_exec_date == 0);
	arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
    }
    iterate_successors_of_BT_reduction_LINEAR_REDUC(eu, context, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
    if ((action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
	dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps->outgoing_mask);
    }
#endif

    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
	struct dague_vp_s **vps = eu->virtual_process->dague_context->virtual_processes;
	data_repo_entry_addto_usage_limit(LINEAR_REDUC_repo, arg.output_entry->key, arg.output_usage);
	for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
	    if (NULL == arg.ready_lists[__vp_id])
		continue;
	    if (__vp_id == eu->virtual_process->vp_id) {
		__dague_schedule(eu, arg.ready_lists[__vp_id]);
	    } else {
		__dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
	    }
	    arg.ready_lists[__vp_id] = NULL;
	}
    }
    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
	int tree_count = context->locals[0].value;
	int i = context->locals[1].value;
	int sz = context->locals[2].value;
	int offset = context->locals[3].value;

	(void) tree_count;
	(void) i;
	(void) sz;
	(void) offset;

	if ((0 == sz)) {
	    data_repo_entry_used_once(eu, REDUCTION_repo, context->data[0].data_repo->key);
	} else if ((0 < sz)) {
	    data_repo_entry_used_once(eu, BT_REDUC_repo, context->data[0].data_repo->key);
	}
	DAGUE_DATA_COPY_RELEASE(context->data[0].data_in);
	if ((tree_count == i)) {
	    data_repo_entry_used_once(eu, LINE_TERMINATOR_repo, context->data[1].data_repo->key);
	} else if ((tree_count < i)) {
	    data_repo_entry_used_once(eu, LINEAR_REDUC_repo, context->data[1].data_repo->key);
	}
	DAGUE_DATA_COPY_RELEASE(context->data[1].data_in);
    }
    return 0;
}

static int data_lookup_of_BT_reduction_LINEAR_REDUC(dague_execution_unit_t * context,
						    dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    int target_device = 0;
    (void) target_device;
    (void) __dague_handle;
    (void) tass;
    (void) context;
    dague_data_copy_t *chunk = NULL;
    data_repo_entry_t *entry = NULL;
    int tree_count = this_task->locals[0].value;
    int i = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;
    (void) chunk;
    (void) entry;

  /** Lookup the input data, and store them in the context if any */
    if (NULL == (chunk = this_task->data[0].data_in)) {	/* flow C */
	entry = NULL;
	if ((0 == sz)) {
	    int REDUCTIONi = tass[0].value = offset;
	    (void) REDUCTIONi;
	    int REDUCTIONt = tass[1].value =
		BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONt;
	    int REDUCTIONli = tass[2].value =
		BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONli;
	    int REDUCTIONsz = tass[3].value =
		BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONsz;
	    entry = data_repo_lookup_entry(REDUCTION_repo, REDUCTION_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* C:LINEAR_REDUC <- A:REDUCTION */
	    ACQUIRE_FLOW(this_task, "C", &BT_reduction_REDUCTION, "A", tass, chunk);
	} else if ((0 < sz)) {
	    int BT_REDUCtree_count = tass[0].value =
		BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCtree_count;
	    int BT_REDUCt = tass[1].value = i;
	    (void) BT_REDUCt;
	    int BT_REDUCsz = tass[2].value =
		BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCsz;
	    int BT_REDUCs = tass[3].value = sz;
	    (void) BT_REDUCs;
	    int BT_REDUClvl = tass[4].value = (BT_REDUCsz - BT_REDUCs);
	    (void) BT_REDUClvl;
	    int BT_REDUCi = tass[5].value = 0;
	    (void) BT_REDUCi;
	    int BT_REDUCoffset = tass[6].value =
		BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCoffset;
	    entry = data_repo_lookup_entry(BT_REDUC_repo, BT_REDUC_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* C:LINEAR_REDUC <- B:BT_REDUC */
	    ACQUIRE_FLOW(this_task, "C", &BT_reduction_BT_REDUC, "B", tass, chunk);
	}
	this_task->data[0].data_in = chunk;	/* flow C */
	this_task->data[0].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if (NULL != chunk)
	this_task->data[0].data_out = dague_data_get_copy(chunk->original, target_device);
    if (NULL == (chunk = this_task->data[1].data_in)) {	/* flow B */
	entry = NULL;
	if ((tree_count == i)) {
	    int LINE_TERMINATORj = tass[0].value = 0;
	    (void) LINE_TERMINATORj;
	    int LINE_TERMINATORtree_count = tass[1].value =
		BT_reduction_inline_c_expr2_line_205((const dague_handle_t *) __dague_handle, tass);
	    (void) LINE_TERMINATORtree_count;
	    int LINE_TERMINATORi = tass[2].value = LINE_TERMINATORtree_count;
	    (void) LINE_TERMINATORi;
	    int LINE_TERMINATORoffset = tass[3].value =
		BT_reduction_inline_c_expr1_line_207((const dague_handle_t *) __dague_handle, tass);
	    (void) LINE_TERMINATORoffset;
	    entry = data_repo_lookup_entry(LINE_TERMINATOR_repo, LINE_TERMINATOR_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* B:LINEAR_REDUC <- T:LINE_TERMINATOR */
	    ACQUIRE_FLOW(this_task, "B", &BT_reduction_LINE_TERMINATOR, "T", tass, chunk);
	} else if ((tree_count < i)) {
	    int LINEAR_REDUCtree_count = tass[0].value =
		BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, tass);
	    (void) LINEAR_REDUCtree_count;
	    int LINEAR_REDUCi = tass[1].value = (i + 1);
	    (void) LINEAR_REDUCi;
	    int LINEAR_REDUCsz = tass[2].value =
		BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, tass);
	    (void) LINEAR_REDUCsz;
	    int LINEAR_REDUCoffset = tass[3].value =
		BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, tass);
	    (void) LINEAR_REDUCoffset;
	    entry = data_repo_lookup_entry(LINEAR_REDUC_repo, LINEAR_REDUC_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* B:LINEAR_REDUC <- C:LINEAR_REDUC */
	    ACQUIRE_FLOW(this_task, "B", &BT_reduction_LINEAR_REDUC, "C", tass, chunk);
	}
	this_task->data[1].data_in = chunk;	/* flow B */
	this_task->data[1].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    this_task->data[1].data_out = dague_data_get_copy(chunk->original, target_device);
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
    this_task->prof_info.desc = (dague_ddesc_t *) __dague_handle->super.dataA;
    this_task->prof_info.id =
	((dague_ddesc_t *) (__dague_handle->super.dataA))->data_key((dague_ddesc_t *) __dague_handle->super.dataA,
								    offset, 0);
#endif /* defined(DAGUE_PROF_TRACE) */
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;
    (void) chunk;
    (void) entry;

    return DAGUE_HOOK_RETURN_DONE;
}

static int hook_of_BT_reduction_LINEAR_REDUC(dague_execution_unit_t * context, dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    (void) context;
    (void) __dague_handle;
    (void) tass;
    int tree_count = this_task->locals[0].value;
    int i = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;

  /** Declare the variables that will hold the data, and all the accounting for each */
    dague_data_copy_t *gC = this_task->data[0].data_in;
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);
    (void) C;
    dague_data_copy_t *gB = this_task->data[1].data_in;
    void *B = DAGUE_DATA_COPY_GET_PTR(gB);
    (void) B;

  /** Update staring simulation date */
#if defined(DAGUE_SIM)
    this_task->sim_exec_date = 0;
    if ((NULL != eC) && (eC->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eC->sim_exec_date;
    if ((NULL != eB) && (eB->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eB->sim_exec_date;
    if (this_task->function->sim_cost_fct != NULL) {
	this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if (context->largest_simulation_date < this_task->sim_exec_date)
	context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
    cache_buf_referenced(context->closest_cache, C);
    cache_buf_referenced(context->closest_cache, B);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              LINEAR_REDUC BODY                                -----*/

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id], this_task);
    int j;

    if (1 == i) {
	assert(0 == offset);
    }

    /* if this is the first task in the chain then "B" is bogus. Ignore it. */
    if (tree_count != i) {
	for (j = 0; j < NB; j++) {
	    REDUCE(B, C, j);
	}
    }

    if (1 == i && 0 == ((__dague_handle->super.super.context)->my_rank))
	printf("%d\n", *TYPE(C));


/*-----                            END OF LINEAR_REDUC BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY) */

    return DAGUE_HOOK_RETURN_DONE;
}

static int complete_hook_of_BT_reduction_LINEAR_REDUC(dague_execution_unit_t * context,
						      dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
#if defined(DISTRIBUTED)
    int tree_count = this_task->locals[0].value;
    int i = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int offset = this_task->locals[3].value;
#endif /* defined(DISTRIBUTED) */
    (void) context;
    (void) __dague_handle;
    this_task->data[0].data_out->version++;
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
			  this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
    (void) tree_count;
    (void) i;
    (void) sz;
    (void) offset;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
			    LINEAR_REDUC_hash(__dague_handle, this_task->locals));
#endif /* defined(DAGUE_PROF_GRAPHER) */
    release_deps_of_BT_reduction_LINEAR_REDUC(context, this_task, DAGUE_ACTION_RELEASE_REMOTE_DEPS | DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_LOCAL_REFS | 0x1,	/* mask of all dep_index */
					      NULL);
    return 0;
}

static int BT_reduction_LINEAR_REDUC_internal_init(__dague_BT_reduction_internal_handle_t * __dague_handle)
{
    dague_dependencies_t *dep = NULL;
    assignment_t assignments[MAX_LOCAL_COUNT];
    (void) assignments;
    int nb_tasks = 0;
    int32_t tree_count, i, sz, offset;
    int32_t i_min = 0x7fffffff;
    int32_t i_max = 0;
    (void) __dague_handle;
    /* First, find the min and max value for each of the dimensions */
    tree_count = BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, assignments);
    assignments[0].value = tree_count;
    for (i = 1; i <= tree_count; i += 1) {
	assignments[1].value = i;
	sz = BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = sz;
	offset = BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, assignments);
	assignments[3].value = offset;
	i_max = dague_imax(i_max, i);
	i_min = dague_imin(i_min, i);
	if (!LINEAR_REDUC_pred(tree_count, i, sz, offset))
	    continue;
	nb_tasks++;
    }

  /**
   * Set the range variables for the collision-free hash-computation
   */
    __dague_handle->LINEAR_REDUC_i_range = (i_max - i_min) + 1;

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    DEBUG3(("Allocating dependencies array for BT_reduction_LINEAR_REDUC_internal_init\n"));
    if (0 != nb_tasks) {
	ALLOCATE_DEP_TRACKING(dep, i_min, i_max, "i", &symb_BT_reduction_LINEAR_REDUC_i, NULL,
			      DAGUE_DEPENDENCIES_FLAG_FINAL);
    }

    AYU_REGISTER_TASK(&BT_reduction_LINEAR_REDUC);
    __dague_handle->super.super.dependencies_array[2] = dep;
    __dague_handle->super.super.nb_local_tasks += nb_tasks;
    return nb_tasks;
}

static const __dague_chore_t __BT_reduction_LINEAR_REDUC_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = hook_of_BT_reduction_LINEAR_REDUC},
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = NULL},		/* End marker */
};

static const dague_function_t BT_reduction_LINEAR_REDUC = {
    .name = "LINEAR_REDUC",
    .function_id = 2,
    .nb_flows = 2,
    .nb_parameters = 1,
    .nb_locals = 4,
    .params = {&symb_BT_reduction_LINEAR_REDUC_i, NULL},
    .locals =
	{&symb_BT_reduction_LINEAR_REDUC_tree_count, &symb_BT_reduction_LINEAR_REDUC_i,
	 &symb_BT_reduction_LINEAR_REDUC_sz, &symb_BT_reduction_LINEAR_REDUC_offset, NULL},
    .data_affinity = affinity_of_BT_reduction_LINEAR_REDUC,
    .initial_data = NULL,
    .final_data = NULL,
    .priority = NULL,
    .in = {&flow_of_BT_reduction_LINEAR_REDUC_for_C, &flow_of_BT_reduction_LINEAR_REDUC_for_B, NULL},
    .out = {&flow_of_BT_reduction_LINEAR_REDUC_for_C, NULL},
    .flags = 0x0 | DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x3,
    .init = (dague_create_function_t *) NULL,
    .key = (dague_functionkey_fn_t *) LINEAR_REDUC_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __BT_reduction_LINEAR_REDUC_chores,
    .iterate_successors = iterate_successors_of_BT_reduction_LINEAR_REDUC,
    .release_deps = release_deps_of_BT_reduction_LINEAR_REDUC,
    .prepare_input = data_lookup_of_BT_reduction_LINEAR_REDUC,
    .prepare_output = NULL,
    .complete_execution = complete_hook_of_BT_reduction_LINEAR_REDUC,
#if defined(DAGUE_SIM)
    .sim_cost_fct = NULL,
#endif
};


/******                                    BT_REDUC                                    ******/

static inline int expr_of_symb_BT_reduction_BT_REDUC_tree_count_fct(const dague_handle_t * __dague_handle_parent,
								    const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_BT_REDUC_tree_count = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_BT_REDUC_tree_count_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_tree_count = {.name = "tree_count",.context_index = 0,.min =
	&expr_of_symb_BT_reduction_BT_REDUC_tree_count,.max = &expr_of_symb_BT_reduction_BT_REDUC_tree_count,.cst_inc =
	0,.expr_inc = NULL,.flags = 0x0 };

static inline int minexpr_of_symb_BT_reduction_BT_REDUC_t_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t minexpr_of_symb_BT_reduction_BT_REDUC_t = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_BT_REDUC_t_fct}
};

static inline int maxexpr_of_symb_BT_reduction_BT_REDUC_t_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int tree_count = assignments[0].value;

    (void) __dague_handle;
    (void) assignments;
    return tree_count;
}

static const expr_t maxexpr_of_symb_BT_reduction_BT_REDUC_t = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_BT_REDUC_t_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_t = {.name = "t",.context_index = 1,.min =
	&minexpr_of_symb_BT_reduction_BT_REDUC_t,.max = &maxexpr_of_symb_BT_reduction_BT_REDUC_t,.cst_inc =
	1,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_BT_REDUC_sz_fct(const dague_handle_t * __dague_handle_parent,
							    const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_BT_REDUC_sz = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_BT_REDUC_sz_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_sz = {.name = "sz",.context_index = 2,.min =
	&expr_of_symb_BT_reduction_BT_REDUC_sz,.max = &expr_of_symb_BT_reduction_BT_REDUC_sz,.cst_inc = 0,.expr_inc =
	NULL,.flags = 0x0 };

static inline int minexpr_of_symb_BT_reduction_BT_REDUC_s_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t minexpr_of_symb_BT_reduction_BT_REDUC_s = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_BT_REDUC_s_fct}
};

static inline int maxexpr_of_symb_BT_reduction_BT_REDUC_s_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return sz;
}

static const expr_t maxexpr_of_symb_BT_reduction_BT_REDUC_s = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_BT_REDUC_s_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_s = {.name = "s",.context_index = 3,.min =
	&minexpr_of_symb_BT_reduction_BT_REDUC_s,.max = &maxexpr_of_symb_BT_reduction_BT_REDUC_s,.cst_inc =
	1,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_BT_REDUC_lvl_fct(const dague_handle_t * __dague_handle_parent,
							     const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (sz - s);
}

static const expr_t expr_of_symb_BT_reduction_BT_REDUC_lvl = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_BT_REDUC_lvl_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_lvl = {.name = "lvl",.context_index = 4,.min =
	&expr_of_symb_BT_reduction_BT_REDUC_lvl,.max = &expr_of_symb_BT_reduction_BT_REDUC_lvl,.cst_inc = 0,.expr_inc =
	NULL,.flags = 0x0 };

static inline int minexpr_of_symb_BT_reduction_BT_REDUC_i_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t minexpr_of_symb_BT_reduction_BT_REDUC_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_BT_REDUC_i_fct}
};

static inline int maxexpr_of_symb_BT_reduction_BT_REDUC_i_fct(const dague_handle_t * __dague_handle_parent,
							      const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr7_line_120((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t maxexpr_of_symb_BT_reduction_BT_REDUC_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_BT_REDUC_i_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_i = {.name = "i",.context_index = 5,.min =
	&minexpr_of_symb_BT_reduction_BT_REDUC_i,.max = &maxexpr_of_symb_BT_reduction_BT_REDUC_i,.cst_inc =
	1,.expr_inc = NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_BT_REDUC_offset_fct(const dague_handle_t * __dague_handle_parent,
								const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_BT_REDUC_offset = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_BT_REDUC_offset_fct}
};
static const symbol_t symb_BT_reduction_BT_REDUC_offset = {.name = "offset",.context_index = 6,.min =
	&expr_of_symb_BT_reduction_BT_REDUC_offset,.max = &expr_of_symb_BT_reduction_BT_REDUC_offset,.cst_inc =
	0,.expr_inc = NULL,.flags = 0x0 };

static inline int affinity_of_BT_reduction_BT_REDUC(dague_execution_context_t * this_task, dague_data_ref_t * ref)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    int tree_count = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int s = this_task->locals[3].value;
    int lvl = this_task->locals[4].value;
    int i = this_task->locals[5].value;
    int offset = this_task->locals[6].value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void) __dague_handle;
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;
    ref->ddesc = (dague_ddesc_t *) __dague_handle->super.dataA;
    /* Compute data key */
    ref->key = ref->ddesc->data_key(ref->ddesc, (offset + (i * 2)), 0);
    return 1;
}

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue_fct(const dague_handle_t *
												  __dague_handle_parent,
												  const assignment_t *
												  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (1 == s);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue_fct(const dague_handle_t *
												__dague_handle_parent,
												const assignment_t *
												assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) __dague_handle;
    (void) assignments;
    return ((offset + (i * 2)) + 1);
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue,
    .ctl_gather_nb = NULL,
    .function_id = 0,		/* BT_reduction_REDUCTION */
    .flow = &flow_of_BT_reduction_REDUCTION_for_A,
    .dep_index = 1,
    .dep_datatype_index = 1,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct(const dague_handle_t
												   *
												   __dague_handle_parent,
												   const assignment_t *
												   assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return !(1 == s);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (s - 1);
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return ((i * 2) + 1);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_B,
    .dep_index = 1,
    .dep_datatype_index = 1,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse,
		    &expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse,
		    &expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct(const dague_handle_t *
											   __dague_handle_parent,
											   const assignment_t *
											   assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return ((sz != s) && (0 == (i % 2)));
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (s + 1);
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return (i / 2);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_A,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131,
		    &expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131,
		    &expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct(const dague_handle_t *
											   __dague_handle_parent,
											   const assignment_t *
											   assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;
    int s = assignments[3].value;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return ((sz != s) && (0 != (i % 2)));
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (s + 1);
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return (i / 2);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_B,
    .dep_index = 1,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132,
		    &expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132,
		    &expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133_fct(const dague_handle_t *
											   __dague_handle_parent,
											   const assignment_t *
											   assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[2].value;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (sz == s);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133_fct(const dague_handle_t *
											 __dague_handle_parent,
											 const assignment_t *
											 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133,
    .ctl_gather_nb = NULL,
    .function_id = 2,		/* BT_reduction_LINEAR_REDUC */
    .flow = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .dep_index = 2,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_B,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133}
};

static const dague_flow_t flow_of_BT_reduction_BT_REDUC_for_B = {
    .name = "B",
    .sym_type = SYM_INOUT,
    .flow_flags = FLOW_ACCESS_RW,
    .flow_index = 0,
    .flow_datatype_mask = 0x1,
    .dep_in = {&flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iftrue,
	       &flow_of_BT_reduction_BT_REDUC_for_B_dep1_atline_130_iffalse},
    .dep_out = {&flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131,
		&flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132,
		&flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue_fct(const dague_handle_t *
												  __dague_handle_parent,
												  const assignment_t *
												  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (1 == s);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue_fct(const dague_handle_t *
												__dague_handle_parent,
												const assignment_t *
												assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;
    int offset = assignments[6].value;

    (void) __dague_handle;
    (void) assignments;
    return (offset + (i * 2));
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue,
    .ctl_gather_nb = NULL,
    .function_id = 0,		/* BT_reduction_REDUCTION */
    .flow = &flow_of_BT_reduction_REDUCTION_for_A,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct(const dague_handle_t
												   *
												   __dague_handle_parent,
												   const assignment_t *
												   assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return !(1 == s);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int s = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (s - 1);
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct(const dague_handle_t *
												 __dague_handle_parent,
												 const assignment_t *
												 assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[5].value;

    (void) __dague_handle;
    (void) assignments;
    return (i * 2);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse_fct}
};

static const dep_t flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_B,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_BT_REDUC_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse,
		    &expr_of_p2_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse,
		    &expr_of_p3_for_flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse}
};

static const dague_flow_t flow_of_BT_reduction_BT_REDUC_for_A = {
    .name = "A",
    .sym_type = SYM_IN,
    .flow_flags = FLOW_ACCESS_READ,
    .flow_index = 1,
    .flow_datatype_mask = 0x0,
    .dep_in = {&flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iftrue,
	       &flow_of_BT_reduction_BT_REDUC_for_A_dep1_atline_128_iffalse},
    .dep_out = {NULL}
};

static void
iterate_successors_of_BT_reduction_BT_REDUC(dague_execution_unit_t * eu, const dague_execution_context_t * this_task,
					    uint32_t action_mask, dague_ontask_function_t * ontask, void *ontask_arg)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    dague_execution_context_t nc;
    dague_dep_data_description_t data;
    int vpid_dst = -1, rank_src = 0, rank_dst = 0;
    int tree_count = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int s = this_task->locals[3].value;
    int lvl = this_task->locals[4].value;
    int i = this_task->locals[5].value;
    int offset = this_task->locals[6].value;
    (void) rank_src;
    (void) rank_dst;
    (void) __dague_handle;
    (void) vpid_dst;
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;
    nc.dague_handle = this_task->dague_handle;
    nc.priority = this_task->priority;
    nc.chore_id = 0;
#if defined(DISTRIBUTED)
    rank_src =
	((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
								 (offset + (i * 2)), 0);
#endif
    if (action_mask & 0x7) {	/* Flow of Data B */
	data.data = this_task->data[0].data_out;
	data.arena = __dague_handle->super.arenas[DAGUE_BT_reduction_DEFAULT_ARENA];
	data.layout = data.arena->opaque_dtt;
	data.count = 1;
	data.displ = 0;
	if (action_mask & 0x1) {
	    if (((sz != s) && (0 == (i % 2)))) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_BT_REDUC.function_id];
		const int BT_REDUC_tree_count =
		    BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = BT_REDUC_tree_count;
		const int BT_REDUC_t = t;
		if ((BT_REDUC_t >= (1)) && (BT_REDUC_t <= (BT_REDUC_tree_count))) {
		    nc.locals[1].value = BT_REDUC_t;
		    const int BT_REDUC_sz =
			BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = BT_REDUC_sz;
		    const int BT_REDUC_s = (s + 1);
		    if ((BT_REDUC_s >= (1)) && (BT_REDUC_s <= (BT_REDUC_sz))) {
			nc.locals[3].value = BT_REDUC_s;
			const int BT_REDUC_lvl = (BT_REDUC_sz - BT_REDUC_s);
			nc.locals[4].value = BT_REDUC_lvl;
			const int BT_REDUC_i = (i / 2);
			if ((BT_REDUC_i >= (0))
			    && (BT_REDUC_i <=
				(BT_reduction_inline_c_expr7_line_120
				 ((const dague_handle_t *) __dague_handle, nc.locals)))) {
			    nc.locals[5].value = BT_REDUC_i;
			    const int BT_REDUC_offset =
				BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle,
								     nc.locals);
			    nc.locals[6].value = BT_REDUC_offset;
#if defined(DISTRIBUTED)
			    rank_dst =
				((dague_ddesc_t *) __dague_handle->super.dataA)->
				rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
					(BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
				vpid_dst =
				    ((dague_ddesc_t *) __dague_handle->super.dataA)->
				    vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
					    (BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    nc.priority = __dague_handle->super.super.priority;
			    RELEASE_DEP_OUTPUT(eu, "B", this_task, "A", &nc, rank_src, rank_dst, &data);
			    if (DAGUE_ITERATE_STOP ==
				ontask(eu, &nc, this_task, &flow_of_BT_reduction_BT_REDUC_for_B_dep2_atline_131, &data,
				       rank_src, rank_dst, vpid_dst, ontask_arg))
				return;
			}
		    }
		}
	    }
	}
	if (action_mask & 0x2) {
	    if (((sz != s) && (0 != (i % 2)))) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_BT_REDUC.function_id];
		const int BT_REDUC_tree_count =
		    BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = BT_REDUC_tree_count;
		const int BT_REDUC_t = t;
		if ((BT_REDUC_t >= (1)) && (BT_REDUC_t <= (BT_REDUC_tree_count))) {
		    nc.locals[1].value = BT_REDUC_t;
		    const int BT_REDUC_sz =
			BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = BT_REDUC_sz;
		    const int BT_REDUC_s = (s + 1);
		    if ((BT_REDUC_s >= (1)) && (BT_REDUC_s <= (BT_REDUC_sz))) {
			nc.locals[3].value = BT_REDUC_s;
			const int BT_REDUC_lvl = (BT_REDUC_sz - BT_REDUC_s);
			nc.locals[4].value = BT_REDUC_lvl;
			const int BT_REDUC_i = (i / 2);
			if ((BT_REDUC_i >= (0))
			    && (BT_REDUC_i <=
				(BT_reduction_inline_c_expr7_line_120
				 ((const dague_handle_t *) __dague_handle, nc.locals)))) {
			    nc.locals[5].value = BT_REDUC_i;
			    const int BT_REDUC_offset =
				BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle,
								     nc.locals);
			    nc.locals[6].value = BT_REDUC_offset;
#if defined(DISTRIBUTED)
			    rank_dst =
				((dague_ddesc_t *) __dague_handle->super.dataA)->
				rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
					(BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
				vpid_dst =
				    ((dague_ddesc_t *) __dague_handle->super.dataA)->
				    vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
					    (BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    nc.priority = __dague_handle->super.super.priority;
			    RELEASE_DEP_OUTPUT(eu, "B", this_task, "B", &nc, rank_src, rank_dst, &data);
			    if (DAGUE_ITERATE_STOP ==
				ontask(eu, &nc, this_task, &flow_of_BT_reduction_BT_REDUC_for_B_dep3_atline_132, &data,
				       rank_src, rank_dst, vpid_dst, ontask_arg))
				return;
			}
		    }
		}
	    }
	}
	if (action_mask & 0x4) {
	    if ((sz == s)) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_LINEAR_REDUC.function_id];
		const int LINEAR_REDUC_tree_count =
		    BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = LINEAR_REDUC_tree_count;
		const int LINEAR_REDUC_i = t;
		if ((LINEAR_REDUC_i >= (1)) && (LINEAR_REDUC_i <= (LINEAR_REDUC_tree_count))) {
		    nc.locals[1].value = LINEAR_REDUC_i;
		    const int LINEAR_REDUC_sz =
			BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = LINEAR_REDUC_sz;
		    const int LINEAR_REDUC_offset =
			BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[3].value = LINEAR_REDUC_offset;
#if defined(DISTRIBUTED)
		    rank_dst =
			((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->
										 super.dataA, LINEAR_REDUC_offset, 0);
		    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
			vpid_dst =
			    ((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->
										     super.dataA, LINEAR_REDUC_offset,
										     0);
		    nc.priority = __dague_handle->super.super.priority;
		    RELEASE_DEP_OUTPUT(eu, "B", this_task, "C", &nc, rank_src, rank_dst, &data);
		    if (DAGUE_ITERATE_STOP ==
			ontask(eu, &nc, this_task, &flow_of_BT_reduction_BT_REDUC_for_B_dep4_atline_133, &data,
			       rank_src, rank_dst, vpid_dst, ontask_arg))
			return;
		}
	    }
	}
    }
    /* Flow of data A has only IN dependencies */
    (void) data;
    (void) nc;
    (void) eu;
    (void) ontask;
    (void) ontask_arg;
    (void) rank_dst;
    (void) action_mask;
}

static int release_deps_of_BT_reduction_BT_REDUC(dague_execution_unit_t * eu, dague_execution_context_t * context,
						 uint32_t action_mask, dague_remote_deps_t * deps)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) context->dague_handle;
    dague_release_dep_fct_arg_t arg;
    int __vp_id;
    arg.action_mask = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
#if defined(DISTRIBUTED)
    arg.remote_deps = deps;
#endif /* defined(DISTRIBUTED) */
    assert(NULL != eu);
    arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);
    (void) __dague_handle;
    (void) deps;
    if (action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY)) {
	arg.output_entry =
	    data_repo_lookup_entry_and_create(eu, BT_REDUC_repo, BT_REDUC_hash(__dague_handle, context->locals));
	arg.output_entry->generator = (void *) context;	/* for AYU */
#if defined(DAGUE_SIM)
	assert(arg.output_entry->sim_exec_date == 0);
	arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
    }
    iterate_successors_of_BT_reduction_BT_REDUC(eu, context, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
    if ((action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
	dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps->outgoing_mask);
    }
#endif

    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
	struct dague_vp_s **vps = eu->virtual_process->dague_context->virtual_processes;
	data_repo_entry_addto_usage_limit(BT_REDUC_repo, arg.output_entry->key, arg.output_usage);
	for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
	    if (NULL == arg.ready_lists[__vp_id])
		continue;
	    if (__vp_id == eu->virtual_process->vp_id) {
		__dague_schedule(eu, arg.ready_lists[__vp_id]);
	    } else {
		__dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
	    }
	    arg.ready_lists[__vp_id] = NULL;
	}
    }
    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
	int tree_count = context->locals[0].value;
	int t = context->locals[1].value;
	int sz = context->locals[2].value;
	int s = context->locals[3].value;
	int lvl = context->locals[4].value;
	int i = context->locals[5].value;
	int offset = context->locals[6].value;

	(void) tree_count;
	(void) t;
	(void) sz;
	(void) s;
	(void) lvl;
	(void) i;
	(void) offset;

	if ((1 == s)) {
	    data_repo_entry_used_once(eu, REDUCTION_repo, context->data[0].data_repo->key);
	} else {
	    data_repo_entry_used_once(eu, BT_REDUC_repo, context->data[0].data_repo->key);
	}
	DAGUE_DATA_COPY_RELEASE(context->data[0].data_in);
	if ((1 == s)) {
	    data_repo_entry_used_once(eu, REDUCTION_repo, context->data[1].data_repo->key);
	} else {
	    data_repo_entry_used_once(eu, BT_REDUC_repo, context->data[1].data_repo->key);
	}
	DAGUE_DATA_COPY_RELEASE(context->data[1].data_in);
    }
    return 0;
}

static int data_lookup_of_BT_reduction_BT_REDUC(dague_execution_unit_t * context, dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    int target_device = 0;
    (void) target_device;
    (void) __dague_handle;
    (void) tass;
    (void) context;
    dague_data_copy_t *chunk = NULL;
    data_repo_entry_t *entry = NULL;
    int tree_count = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int s = this_task->locals[3].value;
    int lvl = this_task->locals[4].value;
    int i = this_task->locals[5].value;
    int offset = this_task->locals[6].value;
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;
    (void) chunk;
    (void) entry;

  /** Lookup the input data, and store them in the context if any */
    if (NULL == (chunk = this_task->data[0].data_in)) {	/* flow B */
	entry = NULL;
	if ((1 == s)) {
	    int REDUCTIONi = tass[0].value = ((offset + (i * 2)) + 1);
	    (void) REDUCTIONi;
	    int REDUCTIONt = tass[1].value =
		BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONt;
	    int REDUCTIONli = tass[2].value =
		BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONli;
	    int REDUCTIONsz = tass[3].value =
		BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONsz;
	    entry = data_repo_lookup_entry(REDUCTION_repo, REDUCTION_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* B:BT_REDUC <- A:REDUCTION */
	    ACQUIRE_FLOW(this_task, "B", &BT_reduction_REDUCTION, "A", tass, chunk);
	} else {
	    int BT_REDUCtree_count = tass[0].value =
		BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCtree_count;
	    int BT_REDUCt = tass[1].value = t;
	    (void) BT_REDUCt;
	    int BT_REDUCsz = tass[2].value =
		BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCsz;
	    int BT_REDUCs = tass[3].value = (s - 1);
	    (void) BT_REDUCs;
	    int BT_REDUClvl = tass[4].value = (BT_REDUCsz - BT_REDUCs);
	    (void) BT_REDUClvl;
	    int BT_REDUCi = tass[5].value = ((i * 2) + 1);
	    (void) BT_REDUCi;
	    int BT_REDUCoffset = tass[6].value =
		BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCoffset;
	    entry = data_repo_lookup_entry(BT_REDUC_repo, BT_REDUC_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* B:BT_REDUC <- B:BT_REDUC */
	    ACQUIRE_FLOW(this_task, "B", &BT_reduction_BT_REDUC, "B", tass, chunk);
	}
	this_task->data[0].data_in = chunk;	/* flow B */
	this_task->data[0].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if (NULL != chunk)
	this_task->data[0].data_out = dague_data_get_copy(chunk->original, target_device);
    if (NULL == (chunk = this_task->data[1].data_in)) {	/* flow A */
	entry = NULL;
	if ((1 == s)) {
	    int REDUCTIONi = tass[0].value = (offset + (i * 2));
	    (void) REDUCTIONi;
	    int REDUCTIONt = tass[1].value =
		BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONt;
	    int REDUCTIONli = tass[2].value =
		BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONli;
	    int REDUCTIONsz = tass[3].value =
		BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, tass);
	    (void) REDUCTIONsz;
	    entry = data_repo_lookup_entry(REDUCTION_repo, REDUCTION_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* A:BT_REDUC <- A:REDUCTION */
	    ACQUIRE_FLOW(this_task, "A", &BT_reduction_REDUCTION, "A", tass, chunk);
	} else {
	    int BT_REDUCtree_count = tass[0].value =
		BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCtree_count;
	    int BT_REDUCt = tass[1].value = t;
	    (void) BT_REDUCt;
	    int BT_REDUCsz = tass[2].value =
		BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCsz;
	    int BT_REDUCs = tass[3].value = (s - 1);
	    (void) BT_REDUCs;
	    int BT_REDUClvl = tass[4].value = (BT_REDUCsz - BT_REDUCs);
	    (void) BT_REDUClvl;
	    int BT_REDUCi = tass[5].value = (i * 2);
	    (void) BT_REDUCi;
	    int BT_REDUCoffset = tass[6].value =
		BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, tass);
	    (void) BT_REDUCoffset;
	    entry = data_repo_lookup_entry(BT_REDUC_repo, BT_REDUC_hash(__dague_handle, tass));
	    chunk = entry->data[0];	/* A:BT_REDUC <- B:BT_REDUC */
	    ACQUIRE_FLOW(this_task, "A", &BT_reduction_BT_REDUC, "B", tass, chunk);
	}
	this_task->data[1].data_in = chunk;	/* flow A */
	this_task->data[1].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    this_task->data[1].data_out = dague_data_get_copy(chunk->original, target_device);
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
    this_task->prof_info.desc = (dague_ddesc_t *) __dague_handle->super.dataA;
    this_task->prof_info.id =
	((dague_ddesc_t *) (__dague_handle->super.dataA))->data_key((dague_ddesc_t *) __dague_handle->super.dataA,
								    (offset + (i * 2)), 0);
#endif /* defined(DAGUE_PROF_TRACE) */
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;
    (void) chunk;
    (void) entry;

    return DAGUE_HOOK_RETURN_DONE;
}

static int hook_of_BT_reduction_BT_REDUC(dague_execution_unit_t * context, dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    (void) context;
    (void) __dague_handle;
    (void) tass;
    int tree_count = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int s = this_task->locals[3].value;
    int lvl = this_task->locals[4].value;
    int i = this_task->locals[5].value;
    int offset = this_task->locals[6].value;
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

  /** Declare the variables that will hold the data, and all the accounting for each */
    dague_data_copy_t *gB = this_task->data[0].data_in;
    void *B = DAGUE_DATA_COPY_GET_PTR(gB);
    (void) B;
    dague_data_copy_t *gA = this_task->data[1].data_in;
    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    (void) A;

  /** Update staring simulation date */
#if defined(DAGUE_SIM)
    this_task->sim_exec_date = 0;
    if ((NULL != eB) && (eB->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eB->sim_exec_date;
    if ((NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eA->sim_exec_date;
    if (this_task->function->sim_cost_fct != NULL) {
	this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if (context->largest_simulation_date < this_task->sim_exec_date)
	context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
    cache_buf_referenced(context->closest_cache, B);
    cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                                BT_REDUC BODY                                  -----*/

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id], this_task);
    int j;

    for (j = 0; j < NB; j++) {
	REDUCE(A, B, j);
    }


/*-----                              END OF BT_REDUC BODY                              -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY) */

    return DAGUE_HOOK_RETURN_DONE;
}

static int complete_hook_of_BT_reduction_BT_REDUC(dague_execution_unit_t * context,
						  dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
#if defined(DISTRIBUTED)
    int tree_count = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int sz = this_task->locals[2].value;
    int s = this_task->locals[3].value;
    int lvl = this_task->locals[4].value;
    int i = this_task->locals[5].value;
    int offset = this_task->locals[6].value;
#endif /* defined(DISTRIBUTED) */
    (void) context;
    (void) __dague_handle;
    this_task->data[0].data_out->version++;
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
			  this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
    (void) tree_count;
    (void) t;
    (void) sz;
    (void) s;
    (void) lvl;
    (void) i;
    (void) offset;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
			    BT_REDUC_hash(__dague_handle, this_task->locals));
#endif /* defined(DAGUE_PROF_GRAPHER) */
    release_deps_of_BT_reduction_BT_REDUC(context, this_task, DAGUE_ACTION_RELEASE_REMOTE_DEPS | DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_LOCAL_REFS | 0x7,	/* mask of all dep_index */
					  NULL);
    return 0;
}

static int BT_reduction_BT_REDUC_internal_init(__dague_BT_reduction_internal_handle_t * __dague_handle)
{
    dague_dependencies_t *dep = NULL;
    assignment_t assignments[MAX_LOCAL_COUNT];
    (void) assignments;
    int nb_tasks = 0;
    int32_t tree_count, t, sz, s, lvl, i, offset;
    int32_t t_min = 0x7fffffff, s_min = 0x7fffffff, i_min = 0x7fffffff;
    int32_t t_max = 0, s_max = 0, i_max = 0;
    (void) __dague_handle;
    int32_t t_start, t_end, t_inc;
    int32_t s_start, s_end, s_inc;
    int32_t i_start, i_end, i_inc;
    /* First, find the min and max value for each of the dimensions */
    tree_count = BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, assignments);
    assignments[0].value = tree_count;
    for (t = 1; t <= tree_count; t += 1) {
	assignments[1].value = t;
	sz = BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = sz;
	for (s = 1; s <= sz; s += 1) {
	    assignments[3].value = s;
	    lvl = (sz - s);
	    assignments[4].value = lvl;
	    for (i = 0;
		 i <= BT_reduction_inline_c_expr7_line_120((const dague_handle_t *) __dague_handle, assignments);
		 i += 1) {
		assignments[5].value = i;
		offset = BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, assignments);
		assignments[6].value = offset;
		t_max = dague_imax(t_max, t);
		t_min = dague_imin(t_min, t);
		s_max = dague_imax(s_max, s);
		s_min = dague_imin(s_min, s);
		i_max = dague_imax(i_max, i);
		i_min = dague_imin(i_min, i);
		if (!BT_REDUC_pred(tree_count, t, sz, s, lvl, i, offset))
		    continue;
		nb_tasks++;
	    }
	}
    }

  /**
   * Set the range variables for the collision-free hash-computation
   */
    __dague_handle->BT_REDUC_t_range = (t_max - t_min) + 1;
    __dague_handle->BT_REDUC_s_range = (s_max - s_min) + 1;
    __dague_handle->BT_REDUC_i_range = (i_max - i_min) + 1;

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    DEBUG3(("Allocating dependencies array for BT_reduction_BT_REDUC_internal_init\n"));
    dep = NULL;
    tree_count = BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, assignments);
    assignments[0].value = tree_count;
    t_start = 1;
    t_end = tree_count;
    t_inc = 1;
    for (t = dague_imax(t_start, t_min); t <= dague_imin(t_end, t_max); t += t_inc) {
	assignments[1].value = t;
	sz = BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = sz;
	s_start = 1;
	s_end = sz;
	s_inc = 1;
	for (s = dague_imax(s_start, s_min); s <= dague_imin(s_end, s_max); s += s_inc) {
	    assignments[3].value = s;
	    lvl = (sz - s);
	    assignments[4].value = lvl;
	    i_start = 0;
	    i_end = BT_reduction_inline_c_expr7_line_120((const dague_handle_t *) __dague_handle, assignments);
	    i_inc = 1;
	    for (i = dague_imax(i_start, i_min); i <= dague_imin(i_end, i_max); i += i_inc) {
		assignments[5].value = i;
		offset = BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle, assignments);
		assignments[6].value = offset;
		if (BT_REDUC_pred(tree_count, t, sz, s, lvl, i, offset)) {
		    /* We did find one! Allocate the dependencies array. */
		    if (dep == NULL) {
			ALLOCATE_DEP_TRACKING(dep, t_min, t_max, "t", &symb_BT_reduction_BT_REDUC_t, NULL,
					      DAGUE_DEPENDENCIES_FLAG_NEXT);
		    }
		    if (dep->u.next[t - t_min] == NULL) {
			ALLOCATE_DEP_TRACKING(dep->u.next[t - t_min], s_min, s_max, "s", &symb_BT_reduction_BT_REDUC_s,
					      dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
		    }
		    if (dep->u.next[t - t_min]->u.next[s - s_min] == NULL) {
			ALLOCATE_DEP_TRACKING(dep->u.next[t - t_min]->u.next[s - s_min], i_min, i_max, "i",
					      &symb_BT_reduction_BT_REDUC_i, dep->u.next[t - t_min],
					      DAGUE_DEPENDENCIES_FLAG_FINAL);
		    }
		}
	    }
	}
    }
    (void) t_start;
    (void) t_end;
    (void) t_inc;
    (void) s_start;
    (void) s_end;
    (void) s_inc;
    (void) i_start;
    (void) i_end;
    (void) i_inc;
    AYU_REGISTER_TASK(&BT_reduction_BT_REDUC);
    __dague_handle->super.super.dependencies_array[1] = dep;
    __dague_handle->super.super.nb_local_tasks += nb_tasks;
    return nb_tasks;
}

static const __dague_chore_t __BT_reduction_BT_REDUC_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = hook_of_BT_reduction_BT_REDUC},
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = NULL},		/* End marker */
};

static const dague_function_t BT_reduction_BT_REDUC = {
    .name = "BT_REDUC",
    .function_id = 1,
    .nb_flows = 2,
    .nb_parameters = 3,
    .nb_locals = 7,
    .params = {&symb_BT_reduction_BT_REDUC_t, &symb_BT_reduction_BT_REDUC_s, &symb_BT_reduction_BT_REDUC_i, NULL},
    .locals =
	{&symb_BT_reduction_BT_REDUC_tree_count, &symb_BT_reduction_BT_REDUC_t, &symb_BT_reduction_BT_REDUC_sz,
	 &symb_BT_reduction_BT_REDUC_s, &symb_BT_reduction_BT_REDUC_lvl, &symb_BT_reduction_BT_REDUC_i,
	 &symb_BT_reduction_BT_REDUC_offset, NULL},
    .data_affinity = affinity_of_BT_reduction_BT_REDUC,
    .initial_data = NULL,
    .final_data = NULL,
    .priority = NULL,
    .in = {&flow_of_BT_reduction_BT_REDUC_for_B, &flow_of_BT_reduction_BT_REDUC_for_A, NULL},
    .out = {&flow_of_BT_reduction_BT_REDUC_for_B, NULL},
    .flags = 0x0 | DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x3,
    .init = (dague_create_function_t *) NULL,
    .key = (dague_functionkey_fn_t *) BT_REDUC_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __BT_reduction_BT_REDUC_chores,
    .iterate_successors = iterate_successors_of_BT_reduction_BT_REDUC,
    .release_deps = release_deps_of_BT_reduction_BT_REDUC,
    .prepare_input = data_lookup_of_BT_reduction_BT_REDUC,
    .prepare_output = NULL,
    .complete_execution = complete_hook_of_BT_reduction_BT_REDUC,
#if defined(DAGUE_SIM)
    .sim_cost_fct = NULL,
#endif
};


/******                                  REDUCTION                                    ******/

static inline int minexpr_of_symb_BT_reduction_REDUCTION_i_fct(const dague_handle_t * __dague_handle_parent,
							       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t minexpr_of_symb_BT_reduction_REDUCTION_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = minexpr_of_symb_BT_reduction_REDUCTION_i_fct}
};

static inline int maxexpr_of_symb_BT_reduction_REDUCTION_i_fct(const dague_handle_t * __dague_handle_parent,
							       const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return (NT - 1);
}

static const expr_t maxexpr_of_symb_BT_reduction_REDUCTION_i = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = maxexpr_of_symb_BT_reduction_REDUCTION_i_fct}
};
static const symbol_t symb_BT_reduction_REDUCTION_i = {.name = "i",.context_index = 0,.min =
	&minexpr_of_symb_BT_reduction_REDUCTION_i,.max = &maxexpr_of_symb_BT_reduction_REDUCTION_i,.cst_inc =
	1,.expr_inc = NULL,.flags = DAGUE_SYMBOL_IS_STANDALONE };

static inline int expr_of_symb_BT_reduction_REDUCTION_t_fct(const dague_handle_t * __dague_handle_parent,
							    const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_REDUCTION_t = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_REDUCTION_t_fct}
};
static const symbol_t symb_BT_reduction_REDUCTION_t = {.name = "t",.context_index = 1,.min =
	&expr_of_symb_BT_reduction_REDUCTION_t,.max = &expr_of_symb_BT_reduction_REDUCTION_t,.cst_inc = 0,.expr_inc =
	NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_REDUCTION_li_fct(const dague_handle_t * __dague_handle_parent,
							     const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_REDUCTION_li = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_REDUCTION_li_fct}
};
static const symbol_t symb_BT_reduction_REDUCTION_li = {.name = "li",.context_index = 2,.min =
	&expr_of_symb_BT_reduction_REDUCTION_li,.max = &expr_of_symb_BT_reduction_REDUCTION_li,.cst_inc = 0,.expr_inc =
	NULL,.flags = 0x0 };

static inline int expr_of_symb_BT_reduction_REDUCTION_sz_fct(const dague_handle_t * __dague_handle_parent,
							     const assignment_t * assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, assignments);
}

static const expr_t expr_of_symb_BT_reduction_REDUCTION_sz = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_symb_BT_reduction_REDUCTION_sz_fct}
};
static const symbol_t symb_BT_reduction_REDUCTION_sz = {.name = "sz",.context_index = 3,.min =
	&expr_of_symb_BT_reduction_REDUCTION_sz,.max = &expr_of_symb_BT_reduction_REDUCTION_sz,.cst_inc = 0,.expr_inc =
	NULL,.flags = 0x0 };

static inline int affinity_of_BT_reduction_REDUCTION(dague_execution_context_t * this_task, dague_data_ref_t * ref)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void) __dague_handle;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
    ref->ddesc = (dague_ddesc_t *) __dague_handle->super.dataA;
    /* Compute data key */
    ref->key = ref->ddesc->data_key(ref->ddesc, i, 0);
    return 1;
}

static inline int initial_data_of_BT_reduction_REDUCTION(dague_execution_context_t * this_task, dague_data_ref_t * refs)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    dague_ddesc_t *__d;
    int __flow_nb = 0;
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void) __dague_handle;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
      /** Flow of A */
    __d = (dague_ddesc_t *) __dague_handle->super.dataA;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, i, 0);
    __flow_nb++;

    return __flow_nb;
}

static inline int expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int i = assignments[0].value;

    (void) __dague_handle;
    (void) assignments;
    return i;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 0;
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101_fct}
};

static const dep_t flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101 = {
    .cond = NULL,
    .ctl_gather_nb = NULL,
    .function_id = -1,		/* BT_reduction_dataA */
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_REDUCTION_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101,
		    &expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct(const dague_handle_t *
											    __dague_handle_parent,
											    const assignment_t *
											    assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int li = assignments[2].value;
    int sz = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return ((sz > 0) && (0 == (li % 2)));
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int li = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return (li / 2);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102_fct}
};

static const dep_t flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_A,
    .dep_index = 0,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_REDUCTION_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102,
		    &expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102,
		    &expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct(const dague_handle_t *
											    __dague_handle_parent,
											    const assignment_t *
											    assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int li = assignments[2].value;
    int sz = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return ((sz > 0) && (0 != (li % 2)));
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int t = assignments[1].value;

    (void) __dague_handle;
    (void) assignments;
    return t;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct}
};

static inline int expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct}
};

static inline int expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int li = assignments[2].value;

    (void) __dague_handle;
    (void) assignments;
    return (li / 2);
}

static const expr_t expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103_fct}
};

static const dep_t flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103,
    .ctl_gather_nb = NULL,
    .function_id = 1,		/* BT_reduction_BT_REDUC */
    .flow = &flow_of_BT_reduction_BT_REDUC_for_B,
    .dep_index = 1,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_REDUCTION_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103,
		    &expr_of_p2_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103,
		    &expr_of_p3_for_flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103}
};

static inline int expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104_fct(const dague_handle_t *
											    __dague_handle_parent,
											    const assignment_t *
											    assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;
    int sz = assignments[3].value;

    (void) __dague_handle;
    (void) assignments;
    return (sz == 0);
}

static const expr_t expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104_fct}
};

static inline int expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104_fct(const dague_handle_t *
											  __dague_handle_parent,
											  const assignment_t *
											  assignments)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) __dague_handle_parent;


    (void) __dague_handle;
    (void) assignments;
    return 1;
}

static const expr_t expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104 = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104_fct}
};

static const dep_t flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104 = {
    .cond = &expr_of_cond_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104,
    .ctl_gather_nb = NULL,
    .function_id = 2,		/* BT_reduction_LINEAR_REDUC */
    .flow = &flow_of_BT_reduction_LINEAR_REDUC_for_C,
    .dep_index = 2,
    .dep_datatype_index = 0,
    .datatype = {.type = {.cst = DAGUE_BT_reduction_DEFAULT_ARENA},
		 .layout = {.fct = NULL},
		 .count = {.cst = 1},
		 .displ = {.cst = 0}
		 },
    .belongs_to = &flow_of_BT_reduction_REDUCTION_for_A,
    .call_params = {
		    &expr_of_p1_for_flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104}
};

static const dague_flow_t flow_of_BT_reduction_REDUCTION_for_A = {
    .name = "A",
    .sym_type = SYM_INOUT,
    .flow_flags = FLOW_ACCESS_RW | FLOW_HAS_IN_DEPS,
    .flow_index = 0,
    .flow_datatype_mask = 0x1,
    .dep_in = {&flow_of_BT_reduction_REDUCTION_for_A_dep1_atline_101},
    .dep_out = {&flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102,
		&flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103,
		&flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104}
};

static void
iterate_successors_of_BT_reduction_REDUCTION(dague_execution_unit_t * eu, const dague_execution_context_t * this_task,
					     uint32_t action_mask, dague_ontask_function_t * ontask, void *ontask_arg)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    dague_execution_context_t nc;
    dague_dep_data_description_t data;
    int vpid_dst = -1, rank_src = 0, rank_dst = 0;
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;
    (void) rank_src;
    (void) rank_dst;
    (void) __dague_handle;
    (void) vpid_dst;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
    nc.dague_handle = this_task->dague_handle;
    nc.priority = this_task->priority;
    nc.chore_id = 0;
#if defined(DISTRIBUTED)
    rank_src =
	((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->super.dataA, i, 0);
#endif
    if (action_mask & 0x7) {	/* Flow of Data A */
	data.data = this_task->data[0].data_out;
	data.arena = __dague_handle->super.arenas[DAGUE_BT_reduction_DEFAULT_ARENA];
	data.layout = data.arena->opaque_dtt;
	data.count = 1;
	data.displ = 0;
	if (action_mask & 0x1) {
	    if (((sz > 0) && (0 == (li % 2)))) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_BT_REDUC.function_id];
		const int BT_REDUC_tree_count =
		    BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = BT_REDUC_tree_count;
		const int BT_REDUC_t = t;
		if ((BT_REDUC_t >= (1)) && (BT_REDUC_t <= (BT_REDUC_tree_count))) {
		    nc.locals[1].value = BT_REDUC_t;
		    const int BT_REDUC_sz =
			BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = BT_REDUC_sz;
		    const int BT_REDUC_s = 1;
		    if ((BT_REDUC_s >= (1)) && (BT_REDUC_s <= (BT_REDUC_sz))) {
			nc.locals[3].value = BT_REDUC_s;
			const int BT_REDUC_lvl = (BT_REDUC_sz - BT_REDUC_s);
			nc.locals[4].value = BT_REDUC_lvl;
			const int BT_REDUC_i = (li / 2);
			if ((BT_REDUC_i >= (0))
			    && (BT_REDUC_i <=
				(BT_reduction_inline_c_expr7_line_120
				 ((const dague_handle_t *) __dague_handle, nc.locals)))) {
			    nc.locals[5].value = BT_REDUC_i;
			    const int BT_REDUC_offset =
				BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle,
								     nc.locals);
			    nc.locals[6].value = BT_REDUC_offset;
#if defined(DISTRIBUTED)
			    rank_dst =
				((dague_ddesc_t *) __dague_handle->super.dataA)->
				rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
					(BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
				vpid_dst =
				    ((dague_ddesc_t *) __dague_handle->super.dataA)->
				    vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
					    (BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    nc.priority = __dague_handle->super.super.priority;
			    RELEASE_DEP_OUTPUT(eu, "A", this_task, "A", &nc, rank_src, rank_dst, &data);
			    if (DAGUE_ITERATE_STOP ==
				ontask(eu, &nc, this_task, &flow_of_BT_reduction_REDUCTION_for_A_dep2_atline_102, &data,
				       rank_src, rank_dst, vpid_dst, ontask_arg))
				return;
			}
		    }
		}
	    }
	}
	if (action_mask & 0x2) {
	    if (((sz > 0) && (0 != (li % 2)))) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_BT_REDUC.function_id];
		const int BT_REDUC_tree_count =
		    BT_reduction_inline_c_expr9_line_115((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = BT_REDUC_tree_count;
		const int BT_REDUC_t = t;
		if ((BT_REDUC_t >= (1)) && (BT_REDUC_t <= (BT_REDUC_tree_count))) {
		    nc.locals[1].value = BT_REDUC_t;
		    const int BT_REDUC_sz =
			BT_reduction_inline_c_expr8_line_117((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = BT_REDUC_sz;
		    const int BT_REDUC_s = 1;
		    if ((BT_REDUC_s >= (1)) && (BT_REDUC_s <= (BT_REDUC_sz))) {
			nc.locals[3].value = BT_REDUC_s;
			const int BT_REDUC_lvl = (BT_REDUC_sz - BT_REDUC_s);
			nc.locals[4].value = BT_REDUC_lvl;
			const int BT_REDUC_i = (li / 2);
			if ((BT_REDUC_i >= (0))
			    && (BT_REDUC_i <=
				(BT_reduction_inline_c_expr7_line_120
				 ((const dague_handle_t *) __dague_handle, nc.locals)))) {
			    nc.locals[5].value = BT_REDUC_i;
			    const int BT_REDUC_offset =
				BT_reduction_inline_c_expr6_line_121((const dague_handle_t *) __dague_handle,
								     nc.locals);
			    nc.locals[6].value = BT_REDUC_offset;
#if defined(DISTRIBUTED)
			    rank_dst =
				((dague_ddesc_t *) __dague_handle->super.dataA)->
				rank_of((dague_ddesc_t *) __dague_handle->super.dataA,
					(BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
				vpid_dst =
				    ((dague_ddesc_t *) __dague_handle->super.dataA)->
				    vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
					    (BT_REDUC_offset + (BT_REDUC_i * 2)), 0);
			    nc.priority = __dague_handle->super.super.priority;
			    RELEASE_DEP_OUTPUT(eu, "A", this_task, "B", &nc, rank_src, rank_dst, &data);
			    if (DAGUE_ITERATE_STOP ==
				ontask(eu, &nc, this_task, &flow_of_BT_reduction_REDUCTION_for_A_dep3_atline_103, &data,
				       rank_src, rank_dst, vpid_dst, ontask_arg))
				return;
			}
		    }
		}
	    }
	}
	if (action_mask & 0x4) {
	    if ((sz == 0)) {
		nc.function = __dague_handle->super.super.functions_array[BT_reduction_LINEAR_REDUC.function_id];
		const int LINEAR_REDUC_tree_count =
		    BT_reduction_inline_c_expr5_line_155((const dague_handle_t *) __dague_handle, nc.locals);
		nc.locals[0].value = LINEAR_REDUC_tree_count;
		const int LINEAR_REDUC_i = 1;
		if ((LINEAR_REDUC_i >= (1)) && (LINEAR_REDUC_i <= (LINEAR_REDUC_tree_count))) {
		    nc.locals[1].value = LINEAR_REDUC_i;
		    const int LINEAR_REDUC_sz =
			BT_reduction_inline_c_expr4_line_157((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[2].value = LINEAR_REDUC_sz;
		    const int LINEAR_REDUC_offset =
			BT_reduction_inline_c_expr3_line_158((const dague_handle_t *) __dague_handle, nc.locals);
		    nc.locals[3].value = LINEAR_REDUC_offset;
#if defined(DISTRIBUTED)
		    rank_dst =
			((dague_ddesc_t *) __dague_handle->super.dataA)->rank_of((dague_ddesc_t *) __dague_handle->
										 super.dataA, LINEAR_REDUC_offset, 0);
		    if ((NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank))
#endif /* DISTRIBUTED */
			vpid_dst =
			    ((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->
										     super.dataA, LINEAR_REDUC_offset,
										     0);
		    nc.priority = __dague_handle->super.super.priority;
		    RELEASE_DEP_OUTPUT(eu, "A", this_task, "C", &nc, rank_src, rank_dst, &data);
		    if (DAGUE_ITERATE_STOP ==
			ontask(eu, &nc, this_task, &flow_of_BT_reduction_REDUCTION_for_A_dep4_atline_104, &data,
			       rank_src, rank_dst, vpid_dst, ontask_arg))
			return;
		}
	    }
	}
    }
    (void) data;
    (void) nc;
    (void) eu;
    (void) ontask;
    (void) ontask_arg;
    (void) rank_dst;
    (void) action_mask;
}

static int release_deps_of_BT_reduction_REDUCTION(dague_execution_unit_t * eu, dague_execution_context_t * context,
						  uint32_t action_mask, dague_remote_deps_t * deps)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(const __dague_BT_reduction_internal_handle_t *) context->dague_handle;
    dague_release_dep_fct_arg_t arg;
    int __vp_id;
    arg.action_mask = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
#if defined(DISTRIBUTED)
    arg.remote_deps = deps;
#endif /* defined(DISTRIBUTED) */
    assert(NULL != eu);
    arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);
    (void) __dague_handle;
    (void) deps;
    if (action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY)) {
	arg.output_entry =
	    data_repo_lookup_entry_and_create(eu, REDUCTION_repo, REDUCTION_hash(__dague_handle, context->locals));
	arg.output_entry->generator = (void *) context;	/* for AYU */
#if defined(DAGUE_SIM)
	assert(arg.output_entry->sim_exec_date == 0);
	arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
    }
    iterate_successors_of_BT_reduction_REDUCTION(eu, context, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
    if ((action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
	dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps->outgoing_mask);
    }
#endif

    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
	struct dague_vp_s **vps = eu->virtual_process->dague_context->virtual_processes;
	data_repo_entry_addto_usage_limit(REDUCTION_repo, arg.output_entry->key, arg.output_usage);
	for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
	    if (NULL == arg.ready_lists[__vp_id])
		continue;
	    if (__vp_id == eu->virtual_process->vp_id) {
		__dague_schedule(eu, arg.ready_lists[__vp_id]);
	    } else {
		__dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
	    }
	    arg.ready_lists[__vp_id] = NULL;
	}
    }
    if (action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
	DAGUE_DATA_COPY_RELEASE(context->data[0].data_in);
    }
    return 0;
}

static int data_lookup_of_BT_reduction_REDUCTION(dague_execution_unit_t * context,
						 dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    int target_device = 0;
    (void) target_device;
    (void) __dague_handle;
    (void) tass;
    (void) context;
    dague_data_copy_t *chunk = NULL;
    data_repo_entry_t *entry = NULL;
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
    (void) chunk;
    (void) entry;

  /** Lookup the input data, and store them in the context if any */
    if (NULL == (chunk = this_task->data[0].data_in)) {	/* flow A */
	entry = NULL;
	chunk = dague_data_get_copy(dataA(i, 0), target_device);
	OBJ_RETAIN(chunk);
	this_task->data[0].data_in = chunk;	/* flow A */
	this_task->data[0].data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if (NULL != chunk)
	this_task->data[0].data_out = dague_data_get_copy(chunk->original, target_device);
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
    this_task->prof_info.desc = (dague_ddesc_t *) __dague_handle->super.dataA;
    this_task->prof_info.id =
	((dague_ddesc_t *) (__dague_handle->super.dataA))->data_key((dague_ddesc_t *) __dague_handle->super.dataA, i,
								    0);
#endif /* defined(DAGUE_PROF_TRACE) */
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
    (void) chunk;
    (void) entry;

    return DAGUE_HOOK_RETURN_DONE;
}

static int hook_of_BT_reduction_REDUCTION(dague_execution_unit_t * context, dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
    assignment_t tass[MAX_PARAM_COUNT];
    (void) context;
    (void) __dague_handle;
    (void) tass;
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;

  /** Declare the variables that will hold the data, and all the accounting for each */
    dague_data_copy_t *gA = this_task->data[0].data_in;
    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    (void) A;

  /** Update staring simulation date */
#if defined(DAGUE_SIM)
    this_task->sim_exec_date = 0;
    if ((NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date))
	this_task->sim_exec_date = eA->sim_exec_date;
    if (this_task->function->sim_cost_fct != NULL) {
	this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if (context->largest_simulation_date < this_task->sim_exec_date)
	context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
    cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                                REDUCTION BODY                                -----*/

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id], this_task);
    *TYPE(A) = i;

/*-----                            END OF REDUCTION BODY                              -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY) */

    return DAGUE_HOOK_RETURN_DONE;
}

static int complete_hook_of_BT_reduction_REDUCTION(dague_execution_unit_t * context,
						   dague_execution_context_t * this_task)
{
    const __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) this_task->dague_handle;
#if defined(DISTRIBUTED)
    int i = this_task->locals[0].value;
    int t = this_task->locals[1].value;
    int li = this_task->locals[2].value;
    int sz = this_task->locals[3].value;
#endif /* defined(DISTRIBUTED) */
    (void) context;
    (void) __dague_handle;
    this_task->data[0].data_out->version++;
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
			  this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
			  this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
    (void) i;
    (void) t;
    (void) li;
    (void) sz;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
			    REDUCTION_hash(__dague_handle, this_task->locals));
#endif /* defined(DAGUE_PROF_GRAPHER) */
    release_deps_of_BT_reduction_REDUCTION(context, this_task, DAGUE_ACTION_RELEASE_REMOTE_DEPS | DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_LOCAL_REFS | 0x7,	/* mask of all dep_index */
					   NULL);
    return 0;
}

static int BT_reduction_REDUCTION_internal_init(__dague_BT_reduction_internal_handle_t * __dague_handle)
{
    dague_dependencies_t *dep = NULL;
    assignment_t assignments[MAX_LOCAL_COUNT];
    (void) assignments;
    int nb_tasks = 0;
    int32_t i, t, li, sz;
    int32_t i_min = 0x7fffffff;
    int32_t i_max = 0;
    (void) __dague_handle;
    /* First, find the min and max value for each of the dimensions */
    for (i = 0; i <= (NT - 1); i += 1) {
	assignments[0].value = i;
	t = BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, assignments);
	assignments[1].value = t;
	li = BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = li;
	sz = BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, assignments);
	assignments[3].value = sz;
	i_max = dague_imax(i_max, i);
	i_min = dague_imin(i_min, i);
	if (!REDUCTION_pred(i, t, li, sz))
	    continue;
	nb_tasks++;
    }

  /**
   * Set the range variables for the collision-free hash-computation
   */
    __dague_handle->REDUCTION_i_range = (i_max - i_min) + 1;

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    DEBUG3(("Allocating dependencies array for BT_reduction_REDUCTION_internal_init\n"));
    if (0 != nb_tasks) {
	ALLOCATE_DEP_TRACKING(dep, i_min, i_max, "i", &symb_BT_reduction_REDUCTION_i, NULL,
			      DAGUE_DEPENDENCIES_FLAG_FINAL);
    }

    AYU_REGISTER_TASK(&BT_reduction_REDUCTION);
    __dague_handle->super.super.dependencies_array[0] = dep;
    __dague_handle->super.super.nb_local_tasks += nb_tasks;
    return nb_tasks;
}

static int BT_reduction_REDUCTION_startup_tasks(dague_context_t * context,
						__dague_BT_reduction_internal_handle_t * __dague_handle,
						dague_execution_context_t ** pready_list)
{
    dague_execution_context_t *new_context, new_context_holder, *new_dynamic_context;
    assignment_t *assignments = NULL;
    int vpid = 0;
    int32_t i = -1, t = -1, li = -1, sz = -1;
    (void) i;
    (void) t;
    (void) li;
    (void) sz;
    new_context = &new_context_holder;
    assignments = new_context->locals;
    /* Parse all the inputs and generate the ready execution tasks */
    new_context->dague_handle = (dague_handle_t *) __dague_handle;
    new_context->function = __dague_handle->super.super.functions_array[BT_reduction_REDUCTION.function_id];
    for (i = 0; i <= (NT - 1); i += 1) {
	assignments[0].value = i;
	assignments[1].value = t =
	    BT_reduction_inline_c_expr12_line_95((const dague_handle_t *) __dague_handle, assignments);
	assignments[2].value = li =
	    BT_reduction_inline_c_expr11_line_96((const dague_handle_t *) __dague_handle, assignments);
	assignments[3].value = sz =
	    BT_reduction_inline_c_expr10_line_97((const dague_handle_t *) __dague_handle, assignments);
	if (!REDUCTION_pred(i, t, li, sz))
	    continue;
	if (NULL != ((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of) {
	    vpid =
		((dague_ddesc_t *) __dague_handle->super.dataA)->vpid_of((dague_ddesc_t *) __dague_handle->super.dataA,
									 i, 0);
	    assert(context->nb_vp >= vpid);
	}
	new_dynamic_context =
	    (dague_execution_context_t *) dague_lifo_pop(&context->virtual_processes[vpid]->execution_units[0]->
							 context_mempool->mempool);
	if (NULL == new_dynamic_context)
	    new_dynamic_context =
		(dague_execution_context_t *) dague_thread_mempool_allocate(context->virtual_processes[0]->
									    execution_units[0]->context_mempool);
	/* Copy only the valid elements from new_context to new_dynamic one */
	new_dynamic_context->dague_handle = new_context->dague_handle;
	new_dynamic_context->function = new_context->function;
	new_dynamic_context->chore_id = 0;
	memcpy(new_dynamic_context->locals, new_context->locals, 4 * sizeof(assignment_t));
	DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
	DAGUE_LIST_ITEM_SINGLETON(new_dynamic_context);
	new_dynamic_context->priority = __dague_handle->super.super.priority;
	new_dynamic_context->data[0].data_repo = NULL;
	new_dynamic_context->data[0].data_in = NULL;
	new_dynamic_context->data[0].data_out = NULL;
#if DAGUE_DEBUG_VERBOSE != 0
	{
	    char tmp[128];
	    DEBUG2(("Add startup task %s\n", dague_snprintf_execution_context(tmp, 128, new_dynamic_context)));
	}
#endif
	dague_dependencies_mark_task_as_startup(new_dynamic_context);
	if (NULL != pready_list[vpid]) {
	    dague_list_item_ring_merge((dague_list_item_t *) new_dynamic_context,
				       (dague_list_item_t *) (pready_list[vpid]));
	}
	pready_list[vpid] = new_dynamic_context;
    }
    return 0;
}

static const __dague_chore_t __BT_reduction_REDUCTION_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = hook_of_BT_reduction_REDUCTION},
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = NULL},		/* End marker */
};

static const dague_function_t BT_reduction_REDUCTION = {
    .name = "REDUCTION",
    .function_id = 0,
    .nb_flows = 1,
    .nb_parameters = 1,
    .nb_locals = 4,
    .params = {&symb_BT_reduction_REDUCTION_i, NULL},
    .locals =
	{&symb_BT_reduction_REDUCTION_i, &symb_BT_reduction_REDUCTION_t, &symb_BT_reduction_REDUCTION_li,
	 &symb_BT_reduction_REDUCTION_sz, NULL},
    .data_affinity = affinity_of_BT_reduction_REDUCTION,
    .initial_data = initial_data_of_BT_reduction_REDUCTION,
    .final_data = NULL,
    .priority = NULL,
    .in = {&flow_of_BT_reduction_REDUCTION_for_A, NULL},
    .out = {&flow_of_BT_reduction_REDUCTION_for_A, NULL},
    .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x1,
    .init = (dague_create_function_t *) NULL,
    .key = (dague_functionkey_fn_t *) REDUCTION_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __BT_reduction_REDUCTION_chores,
    .iterate_successors = iterate_successors_of_BT_reduction_REDUCTION,
    .release_deps = release_deps_of_BT_reduction_REDUCTION,
    .prepare_input = data_lookup_of_BT_reduction_REDUCTION,
    .prepare_output = NULL,
    .complete_execution = complete_hook_of_BT_reduction_REDUCTION,
#if defined(DAGUE_SIM)
    .sim_cost_fct = NULL,
#endif
};


static const dague_function_t *BT_reduction_functions[] = {
    &BT_reduction_REDUCTION,
    &BT_reduction_BT_REDUC,
    &BT_reduction_LINEAR_REDUC,
    &BT_reduction_LINE_TERMINATOR
};

static void BT_reduction_startup(dague_context_t * context, dague_handle_t * dague_handle,
				 dague_execution_context_t ** pready_list)
{
    uint32_t supported_dev = 0;
    __dague_BT_reduction_internal_handle_t *__dague_handle = (__dague_BT_reduction_internal_handle_t *) dague_handle;
    dague_handle->context = context;
    /* Create the PINS DATA pointers if PINS is enabled */
#  if defined(PINS_ENABLE)
    __dague_handle->super.super.context = context;
    (void) pins_handle_init(&__dague_handle->super.super);
#  endif /* defined(PINS_ENABLE) */

    uint32_t wanted_devices = dague_handle->devices_mask;
    dague_handle->devices_mask = 0;
    uint32_t _i;
    for (_i = 0; _i < dague_nb_devices; _i++) {
	if (!(wanted_devices & (1 << _i)))
	    continue;
	dague_device_t *device = dague_devices_get(_i);
	dague_ddesc_t *dague_ddesc;

	if (NULL == device)
	    continue;
	if (NULL != device->device_handle_register)
	    if (DAGUE_SUCCESS != device->device_handle_register(device, (dague_handle_t *) dague_handle))
		continue;

	if (NULL != device->device_memory_register) {	/* Register all the data */
	    dague_ddesc = (dague_ddesc_t *) __dague_handle->super.dataA;
	    if (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) {
		continue;
	    }
	}
	supported_dev |= (1 << device->type);
	dague_handle->devices_mask |= (1 << _i);
    }
    /* Remove all the chores without a backend device */
    uint32_t i;
    for (i = 0; i < dague_handle->nb_functions; i++) {
	dague_function_t *func = (dague_function_t *) dague_handle->functions_array[i];
	__dague_chore_t *chores = (__dague_chore_t *) func->incarnations;
	uint32_t index = 0;
	uint32_t j;
	for (j = 0; NULL != chores[j].hook; j++) {
	    if (supported_dev & (1 << chores[j].type)) {
		if (j != index)
		    chores[index] = chores[j];
		index++;
	    }
	}
	chores[index].type = DAGUE_DEV_NONE;
	chores[index].evaluate = NULL;
	chores[index].hook = NULL;
    }
    BT_reduction_LINE_TERMINATOR_startup_tasks(context, (__dague_BT_reduction_internal_handle_t *) dague_handle,
					       pready_list);
    BT_reduction_REDUCTION_startup_tasks(context, (__dague_BT_reduction_internal_handle_t *) dague_handle, pready_list);
}

static void BT_reduction_destructor(__dague_BT_reduction_internal_handle_t * handle)
{
    uint32_t i;
    for (i = 0; i < handle->super.super.nb_functions; i++) {
	dague_function_t *func = (dague_function_t *) handle->super.super.functions_array[i];
	free((void *) func->incarnations);
	free(func);
    }
    free(handle->super.super.functions_array);
    handle->super.super.functions_array = NULL;
    handle->super.super.nb_functions = 0;
    for (i = 0; i < (uint32_t) handle->super.arenas_size; i++) {
	if (handle->super.arenas[i] != NULL) {
	    dague_arena_destruct(handle->super.arenas[i]);
	    free(handle->super.arenas[i]);
	    handle->super.arenas[i] = NULL;
	}
    }
    free(handle->super.arenas);
    handle->super.arenas = NULL;
    handle->super.arenas_size = 0;
    /* Destroy the data repositories for this object */
    data_repo_destroy_nothreadsafe(handle->LINE_TERMINATOR_repository);
    data_repo_destroy_nothreadsafe(handle->LINEAR_REDUC_repository);
    data_repo_destroy_nothreadsafe(handle->BT_REDUC_repository);
    data_repo_destroy_nothreadsafe(handle->REDUCTION_repository);
    for (i = 0; i < DAGUE_BT_reduction_NB_FUNCTIONS; i++) {
	dague_destruct_dependencies(handle->super.super.dependencies_array[i]);
	handle->super.super.dependencies_array[i] = NULL;
    }
    free(handle->super.super.dependencies_array);
    handle->super.super.dependencies_array = NULL;
    /* Unregister all the data */
    uint32_t _i;
    for (_i = 0; _i < dague_nb_devices; _i++) {
	dague_device_t *device;
	dague_ddesc_t *dague_ddesc;
	if (!(handle->super.super.devices_mask & (1 << _i)))
	    continue;
	if ((NULL == (device = dague_devices_get(_i))) || (NULL == device->device_memory_unregister))
	    continue;
	dague_ddesc = (dague_ddesc_t *) handle->super.dataA;
	(void) dague_ddesc->unregister_memory(dague_ddesc, device);
    }
    /* Unregister the handle from the devices */
    for (i = 0; i < dague_nb_devices; i++) {
	if (!(handle->super.super.devices_mask & (1 << i)))
	    continue;
	handle->super.super.devices_mask ^= (1 << i);
	dague_device_t *device = dague_devices_get(i);
	if ((NULL == device) || (NULL == device->device_handle_unregister))
	    continue;
	if (DAGUE_SUCCESS != device->device_handle_unregister(device, &handle->super.super))
	    continue;
    }
    dague_handle_unregister(&handle->super.super);
    free(handle);
}

#undef dataA
#undef NB
#undef NT

dague_BT_reduction_handle_t *dague_BT_reduction_new(struct tiled_matrix_desc_t *dataA /* data dataA */ , int NB, int NT)
{
    __dague_BT_reduction_internal_handle_t *__dague_handle =
	(__dague_BT_reduction_internal_handle_t *) calloc(1, sizeof(__dague_BT_reduction_internal_handle_t));
    /* Dump the hidden parameters */

    int i, j;
    int LINE_TERMINATOR_nblocal_tasks;
    int LINEAR_REDUC_nblocal_tasks;
    int BT_REDUC_nblocal_tasks;
    int REDUCTION_nblocal_tasks;

    __dague_handle->super.super.nb_functions = DAGUE_BT_reduction_NB_FUNCTIONS;
    __dague_handle->super.super.dependencies_array = (dague_dependencies_t **)
	calloc(DAGUE_BT_reduction_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    __dague_handle->super.super.devices_mask = DAGUE_DEVICES_ALL;
    __dague_handle->super.super.functions_array =
	(const dague_function_t **) malloc(DAGUE_BT_reduction_NB_FUNCTIONS * sizeof(dague_function_t *));
    for (i = 0; i < (int) __dague_handle->super.super.nb_functions; i++) {
	dague_function_t *func;
	__dague_handle->super.super.functions_array[i] = malloc(sizeof(dague_function_t));
	memcpy((dague_function_t *) __dague_handle->super.super.functions_array[i], BT_reduction_functions[i],
	       sizeof(dague_function_t));
	func = (dague_function_t *) __dague_handle->super.super.functions_array[i];
	for (j = 0; NULL != func->incarnations[j].hook; j++);
	func->incarnations = (__dague_chore_t *) malloc((j + 1) * sizeof(__dague_chore_t));
	memcpy((__dague_chore_t *) func->incarnations, BT_reduction_functions[i]->incarnations,
	       (j + 1) * sizeof(__dague_chore_t));
    }
    /* Compute the number of arenas: */
    /*   DAGUE_BT_reduction_DEFAULT_ARENA  ->  0 */
    __dague_handle->super.arenas_size = 1;
    __dague_handle->super.arenas =
	(dague_arena_t **) malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t *));
    for (i = 0; i < __dague_handle->super.arenas_size; i++) {
	__dague_handle->super.arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }
    /* Now the Parameter-dependent structures: */
    __dague_handle->super.dataA = dataA;
    __dague_handle->super.NB = NB;
    __dague_handle->super.NT = NT;
    /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
    __dague_handle->super.super.profiling_array = BT_reduction_profiling_array;
    if (-1 == BT_reduction_profiling_array[0]) {
	dague_profiling_add_dictionary_keyword("LINE_TERMINATOR", "fill:CC2828",
					       sizeof(dague_profile_ddesc_info_t) + 4 * sizeof(assignment_t),
					       dague_profile_ddesc_key_to_string,
					       (int *) &__dague_handle->super.super.profiling_array[0 +
												    2 *
												    BT_reduction_LINE_TERMINATOR.
												    function_id
												    /* LINE_TERMINATOR start key */
												    ],
					       (int *) &__dague_handle->super.super.profiling_array[1 +
												    2 *
												    BT_reduction_LINE_TERMINATOR.
												    function_id
												    /* LINE_TERMINATOR end key */
												    ]);
	dague_profiling_add_dictionary_keyword("LINEAR_REDUC", "fill:7ACC28",
					       sizeof(dague_profile_ddesc_info_t) + 4 * sizeof(assignment_t),
					       dague_profile_ddesc_key_to_string,
					       (int *) &__dague_handle->super.super.profiling_array[0 +
												    2 *
												    BT_reduction_LINEAR_REDUC.
												    function_id
												    /* LINEAR_REDUC start key */
												    ],
					       (int *) &__dague_handle->super.super.profiling_array[1 +
												    2 *
												    BT_reduction_LINEAR_REDUC.
												    function_id
												    /* LINEAR_REDUC end key */
												    ]);
	dague_profiling_add_dictionary_keyword("BT_REDUC", "fill:28CCCC",
					       sizeof(dague_profile_ddesc_info_t) + 7 * sizeof(assignment_t),
					       dague_profile_ddesc_key_to_string,
					       (int *) &__dague_handle->super.super.profiling_array[0 +
												    2 *
												    BT_reduction_BT_REDUC.
												    function_id
												    /* BT_REDUC start key */
												    ],
					       (int *) &__dague_handle->super.super.profiling_array[1 +
												    2 *
												    BT_reduction_BT_REDUC.
												    function_id
												    /* BT_REDUC end key */
												    ]);
	dague_profiling_add_dictionary_keyword("REDUCTION", "fill:7A28CC",
					       sizeof(dague_profile_ddesc_info_t) + 4 * sizeof(assignment_t),
					       dague_profile_ddesc_key_to_string,
					       (int *) &__dague_handle->super.super.profiling_array[0 +
												    2 *
												    BT_reduction_REDUCTION.
												    function_id
												    /* REDUCTION start key */
												    ],
					       (int *) &__dague_handle->super.super.profiling_array[1 +
												    2 *
												    BT_reduction_REDUCTION.
												    function_id
												    /* REDUCTION end key */
												    ]);
    }
#  endif /* defined(DAGUE_PROF_TRACE) */
    /* Create the data repositories for this object */
    LINE_TERMINATOR_nblocal_tasks = BT_reduction_LINE_TERMINATOR_internal_init(__dague_handle);
    __dague_handle->LINE_TERMINATOR_repository = data_repo_create_nothreadsafe(LINE_TERMINATOR_nblocal_tasks, 1);

    LINEAR_REDUC_nblocal_tasks = BT_reduction_LINEAR_REDUC_internal_init(__dague_handle);
    __dague_handle->LINEAR_REDUC_repository = data_repo_create_nothreadsafe(LINEAR_REDUC_nblocal_tasks, 2);

    BT_REDUC_nblocal_tasks = BT_reduction_BT_REDUC_internal_init(__dague_handle);
    __dague_handle->BT_REDUC_repository = data_repo_create_nothreadsafe(BT_REDUC_nblocal_tasks, 2);

    REDUCTION_nblocal_tasks = BT_reduction_REDUCTION_internal_init(__dague_handle);
    __dague_handle->REDUCTION_repository = data_repo_create_nothreadsafe(REDUCTION_nblocal_tasks, 1);

    __dague_handle->super.super.startup_hook = BT_reduction_startup;
    __dague_handle->super.super.destructor = (dague_destruct_fn_t) BT_reduction_destructor;
    (void) dague_handle_register((dague_handle_t *) __dague_handle);
    return (dague_BT_reduction_handle_t *) __dague_handle;
}
