#line 1 "gemm.jdf"
   /**
    * PLASMA include for defined and constants.
    */
#include <plasma.h>
#include <core_dblas.h>

extern PLASMA_desc descA, descB, descC;
extern int uplo;
int PLASMA_INFO;

#include "dplasma.h"
#include "remote_dep.h"

#ifdef DISTRIBUTED
#   include "data_management.h"
    extern DPLASMA_desc ddescA, ddescB, ddescC;
#   define A(m,n) dplasma_get_local_tile_s(&ddescA, m, n)
#   define B(m,n) dplasma_get_local_tile_s(&ddescB, m, n)
#   define C(m,n) dplasma_get_local_tile_s(&ddescC, m, n)
#else
#   define A(m,n) &(((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)])
#   define B(m,n) &(((double*)descB.mat)[descB.bsiz*(m)+descB.bsiz*descB.lmt*(n)])
#   define C(m,n) &(((double*)descC.mat)[descC.bsiz*(m)+descC.bsiz*descC.lmt*(n)])
#endif

#ifdef DPLASMA_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define OUTPUT(ARG)  printf ARG
#else
#   define OUTPUT(ARG)
#endif

#ifdef DPLASMA_DRY_RUN
#   define CORE(FN, ARGS)
#else
#   define CORE(FN, ARGS) CORE_##FN ARGS
#endif

#line 42 "dgemm.c"
static expr_t expr0 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr1 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static symbol_t symb0 = { .flags = 0x00000001, .name = "GRIDrows", .min = &expr0, .max = &expr1 };
static expr_t expr2 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr3 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static symbol_t symb1 = { .flags = 0x00000001, .name = "GRIDcols", .min = &expr2, .max = &expr3 };
static expr_t expr4 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 120 }; /* 120 */
static expr_t expr5 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 120 }; /* 120 */
static symbol_t symb2 = { .flags = 0x00000001, .name = "NB", .min = &expr4, .max = &expr5 };
static symbol_t symb3 = { .flags = 0x00000001, .name = "SIZE", .min = NULL, .max = NULL };
static expr_t expr6 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr7 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static symbol_t symb4 = { .flags = 0x00000001, .name = "colRANK", .min = &expr6, .max = &expr7 };
static expr_t expr8 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr9 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static symbol_t symb5 = { .flags = 0x00000001, .name = "rowRANK", .min = &expr8, .max = &expr9 };
static expr_t expr10 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr11 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static symbol_t symb6 = { .flags = 0x00000001, .name = "stileSIZE", .min = &expr10, .max = &expr11 };
static symbol_t *dplasma_symbols[] = {
   &symb0,
   &symb1,
   &symb2,
   &symb3,
   &symb4,
   &symb5,
   &symb6};

int GRIDrows = 1;
int GRIDcols = 1;
int NB = 120;
int SIZE;
int colRANK = 0;
int rowRANK = 0;
int stileSIZE = 1;

#include <assert.h>
#include <string.h>
#include "remote_dep.h"
#include "datarepo.h"

#define TILE_SIZE (DPLASMA_TILE_SIZE*DPLASMA_TILE_SIZE*sizeof(double))
#ifdef HAVE_PAPI
#include "papi.h"
extern int eventSet;
#endif

#if defined(DPLASMA_GRAPHER)
#include <stdio.h>
extern FILE *__dplasma_graph_file;
#define COLORS_SIZE 54
static char *colors[54] = {
  "#E52B50",
  "#7FFFD4",
  "#007FFF",
  "#000000",
  "#0000FF",
  "#0095B6",
  "#8A2BE2",
  "#A52A2A",
  "#702963",
  "#960018",
  "#DE3163",
  "#007BA7",
  "#7FFF00",
  "#F88379",
  "#DC143C",
  "#00FFFF",
  "#7DF9FF",
  "#FFD700",
  "#808080",
  "#00CC00",
  "#3FFF00",
  "#4B0082",
  "#00A86B",
  "#B57EDC",
  "#C8A2C8",
  "#BFFF00",
  "#FF00FF",
  "#800000",
  "#E0B0FF",
  "#000080",
  "#808000",
  "#FFA500",
  "#FF4500",
  "#FFE5B4",
  "#1C39BB",
  "#FFC0CB",
  "#843179",
  "#FF7518",
  "#800080",
  "#FF0000",
  "#C71585",
  "#FF007F",
  "#FA8072",
  "#FF2400",
  "#C0C0C0",
  "#708090",
  "#00FF7F",
  "#483C32",
  "#008080",
  "#40E0D0",
  "#EE82EE",
  "#40826D",
  "#FFFF00",
  "(null)"
};
#endif /* defined(DPLASMA_GRAPHER) */
#ifdef DPLASMA_PROFILING
#include "profiling.h"
int STARTUP_start_key, STARTUP_end_key;
int A_start_key, A_end_key;
int READVALS_start_key, READVALS_end_key;
int GEMM_start_key, GEMM_end_key;
int B_start_key, B_end_key;
int C_start_key, C_end_key;
#define TAKE_TIME(EU_CONTEXT, KEY, ID)  dplasma_profiling_trace((EU_CONTEXT)->eu_profile, (KEY), (ID))
#else
#define TAKE_TIME(EU_CONTEXT, KEY, ID)
#endif  /* DPLASMA_PROFILING */

#include "scheduling.h"

static long int STARTUP_hash(int useless){
  return 0+ ( (useless-(0))* 1);
}

static data_repo_t *STARTUP_repo = NULL;
static long int READVALS_hash(int i, int j){
  return 0+ ( (i-(0))* 1)+ ( (j-(0))* 1* (((SIZE)-(1))+1-(0)));
}

static data_repo_t *READVALS_repo = NULL;
static long int GEMM_hash(int i, int j, int k){
  return 0+ ( (i-(0))* 1)+ ( (j-(0))* 1* (((SIZE)-(1))+1-(0)))+ ( (k-(0))* 1* (((SIZE)-(1))+1-(0))* (((SIZE)-(1))+1-(0)));
}

static data_repo_t *GEMM_repo = NULL;
static expr_t expr12 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr13 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static symbol_t symb7 = { .flags = 0x00000002, .name = "useless", .min = &expr12, .max = &expr13 };
static expr_t expr14 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr15 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static dep_t dep0 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr14, &expr15, NULL, NULL, NULL}};
static expr_t expr17 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr19 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr20 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr18 = { .op = 25, .flags = 0, .bop1 = &expr19, .bop2 = &expr20, .value = 0 }; /* SIZE - 1 */
static expr_t expr16 = { .op = 24, .flags = 0, .bop1 = &expr17, .bop2 = &expr18, .value = 0 }; /*  [0 .. SIZE - 1]  */
static expr_t expr22 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr24 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr25 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr23 = { .op = 25, .flags = 0, .bop1 = &expr24, .bop2 = &expr25, .value = 0 }; /* SIZE - 1 */
static expr_t expr21 = { .op = 24, .flags = 0, .bop1 = &expr22, .bop2 = &expr23, .value = 0 }; /*  [0 .. SIZE - 1]  */
static dep_t dep1 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr16, &expr21, NULL, NULL, NULL}};
static param_t param0 = { .name = "USELESS", .sym_type = 3, .param_mask = 0x01,
     .dep_in  = {&dep0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static int STARTUP_release_dependencies(dplasma_execution_unit_t *context,
                                   const dplasma_execution_context_t *exec_context,
                                   int propagate_remote_dep,
                                   void **data)
{
  int ret = 0;
  data_repo_entry_t *eSTARTUP;
  int useless = exec_context->locals[0].value;
  useless = exec_context->locals[0].value;
  eSTARTUP = data_repo_lookup_entry( STARTUP_repo, STARTUP_hash(useless), 1 );
  if(data) {
    eSTARTUP->data[0] = data[0];
  } else {
    eSTARTUP->data[0] = NULL;
  }

  dplasma_execution_context_t*   ready_list = NULL;
  uint32_t usage = 0;
  dplasma_execution_context_t new_context = { .function = NULL, .locals = { {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL}}};
  /* remove warnings about unused context variable*/
  (void)context;
  /* remove warnings in case the variable is not used later */
  (void)useless;
#ifdef DISTRIBUTED
  if(propagate_remote_dep) {
    dplasma_remote_dep_reset_forwarded(context);
  }
#else
  (void)propagate_remote_dep;  /* silence a warning */
#endif
  new_context.function = exec_context->function->inout[0]->dep_out[0]->dplasma; /* READVALS */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    assert( strcmp( exec_context->function->inout[0]->dep_out[0]->dplasma->name, "READVALS") == 0 );
    for(_p0 = 0; _p0 <= (SIZE)-(1); _p0++) {
      for(_p1 = 0; _p1 <= (SIZE)-(1); _p1++) {
         {
          int i = _p0;
          int j = _p1;
          (void)i;
          (void)j;
          if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
            struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
            struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
            new_context.locals[0].sym = new_context.function->locals[0]; /* task READVALS */
            new_context.locals[0].value = _p0;  /* task READVALS local i */
            new_context.locals[1].sym = new_context.function->locals[1]; /* task READVALS */
            new_context.locals[1].value = _p1;  /* task READVALS local j */
            usage++;
            ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                           exec_context->function->inout[0/*i*/],
                           &new_context,
                           exec_context->function->inout[0/*i*/]->dep_out[0/*j*/]->param,
                           j_placeholder, &ready_list);
          } else if (propagate_remote_dep) {
            int rank, rrank, crank, ncols;
            rrank = ((i)/(stileSIZE))%(GRIDrows);
            crank = ((j)/(stileSIZE))%(GRIDcols);
            ncols = GRIDcols;
            rank = crank + rrank * ncols;
          //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
            ret += dplasma_remote_dep_activate_rank(context,
                           exec_context,
                           exec_context->function->inout[0/*i*/],
                           rank, data);
          }
        }
      }
    }
  }
  data_repo_entry_set_usage_limit(STARTUP_repo, eSTARTUP->key, usage);
  if( NULL != ready_list )
    __dplasma_schedule(context, ready_list);
  return ret;
}
static int STARTUP_hook(dplasma_execution_unit_t* context, dplasma_execution_context_t *exec_context)
{
  (void)context;
  int useless = exec_context->locals[0].value;
  useless = exec_context->locals[0].value;
  void *USELESS = NULL;
  data_repo_entry_t *eUSELESS = NULL;
  /* remove warnings in case the variable is not used later */
  (void)useless;
  USELESS = A(0, 0);


#ifdef HAVE_PAPI
  int i, num_events;
  int events[MAX_EVENTS];
  PAPI_list_events(eventSet, &events, &num_events);
  long long values[num_events];
  PAPI_start(eventSet);
#endif

#if defined(DPLASMA_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, USELESS);
#endif /* DPLASMA_CACHE_AWARENESS */
  TAKE_TIME(context, STARTUP_start_key, STARTUP_hash(useless));
  #line 63 "gemm.jdf"
  /* Nothing relevant */

#line 306 "dgemm.c"

  TAKE_TIME(context, STARTUP_end_key, STARTUP_hash(useless));

#ifdef HAVE_PAPI
  PAPI_stop(eventSet, &values);
  if(num_events > 0) {
    printf("PAPI counter values from STARTUP (thread=%ld): ", context->eu_id);
    for(i=0; i<num_events; ++i) {
      char event_name[PAPI_MAX_STR_LEN];
      PAPI_event_code_to_name(events[i], &event_name);
      printf("   %s  %lld ", event_name, values[i]);
    }
    printf("\n");
  }
#endif

#if defined(DPLASMA_GRAPHER)
if( NULL != __dplasma_graph_file ) {
  char tmp[128];
  dplasma_service_to_string(exec_context, tmp, 128);
  fprintf(__dplasma_graph_file,
          "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\",tooltip=\"STARTUP%ld\"];\n",
          tmp, colors[context->eu_id], tmp, STARTUP_hash(useless));
}
#endif /* defined(DPLASMA_GRAPHER) */
  {
    void *data[1];
    data[0] = USELESS;
    STARTUP_release_dependencies(context, exec_context, 1, data);
  }
  return 0;
}

static expr_t expr26 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr28 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr29 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr27 = { .op = 25, .flags = 0, .bop1 = &expr28, .bop2 = &expr29, .value = 0 }; /* SIZE - 1 */
static symbol_t symb8 = { .flags = 0x00000002, .name = "i", .min = &expr26, .max = &expr27 };
static expr_t expr30 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr32 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr33 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr31 = { .op = 25, .flags = 0, .bop1 = &expr32, .bop2 = &expr33, .value = 0 }; /* SIZE - 1 */
static symbol_t symb9 = { .flags = 0x00000002, .name = "j", .min = &expr30, .max = &expr31 };
static int inline_expr0( const  assignment_t *assignments )
{
  int i = assignments[0].value;
  return (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
}
static expr_t inline0 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr0 };
static int inline_expr1( const  assignment_t *assignments )
{
  int j = assignments[1].value;
  return (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
}
static expr_t inline1 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr1 };
static expr_t expr34 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static dep_t dep2 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr34, NULL, NULL, NULL, NULL}};
static param_t param1 = { .name = "USELESS", .sym_type = 1, .param_mask = 0x01,
     .dep_in  = {&dep2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static expr_t expr35 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr36 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static dep_t dep3 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr35, &expr36, NULL, NULL, NULL}};
static expr_t expr37 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr38 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static expr_t expr39 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static dep_t dep4 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr37, &expr38, &expr39, NULL, NULL}};
static param_t param2 = { .name = "A", .sym_type = 3, .param_mask = 0x02,
     .dep_in  = {&dep3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static expr_t expr40 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr41 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static dep_t dep5 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr40, &expr41, NULL, NULL, NULL}};
static expr_t expr42 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr43 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static expr_t expr44 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static dep_t dep6 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr42, &expr43, &expr44, NULL, NULL}};
static param_t param4 = { .name = "B", .sym_type = 3, .param_mask = 0x04,
     .dep_in  = {&dep5, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep6, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static expr_t expr45 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr46 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static dep_t dep7 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr45, &expr46, NULL, NULL, NULL}};
static expr_t expr47 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb8 }; /* i */
static expr_t expr48 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb9 }; /* j */
static expr_t expr49 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static dep_t dep8 = { .cond = NULL, .dplasma = NULL,
                       .call_params = {&expr47, &expr48, &expr49, NULL, NULL}};
static param_t param6 = { .name = "C", .sym_type = 3, .param_mask = 0x08,
     .dep_in  = {&dep7, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static int READVALS_release_dependencies(dplasma_execution_unit_t *context,
                                   const dplasma_execution_context_t *exec_context,
                                   int propagate_remote_dep,
                                   void **data)
{
  int ret = 0;
  data_repo_entry_t *eREADVALS;
  int i = exec_context->locals[0].value;
  i = exec_context->locals[0].value;
  int j = exec_context->locals[1].value;
  j = exec_context->locals[1].value;
  eREADVALS = data_repo_lookup_entry( READVALS_repo, READVALS_hash(i, j), 1 );
  if(data) {
    eREADVALS->data[0] = data[0];
    eREADVALS->data[1] = data[1];
    eREADVALS->data[2] = data[2];
  } else {
    eREADVALS->data[0] = NULL;
    eREADVALS->data[1] = NULL;
    eREADVALS->data[2] = NULL;
  }

  dplasma_execution_context_t*   ready_list = NULL;
  uint32_t usage = 0;
  dplasma_execution_context_t new_context = { .function = NULL, .locals = { {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL}}};
  /* remove warnings about unused context variable*/
  (void)context;
  /* remove warnings in case the variable is not used later */
  (void)i;
  (void)j;
#ifdef DISTRIBUTED
  if(propagate_remote_dep) {
    dplasma_remote_dep_reset_forwarded(context);
  }
#else
  (void)propagate_remote_dep;  /* silence a warning */
#endif
  new_context.function = exec_context->function->inout[1]->dep_out[0]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[1]->dep_out[0]->dplasma->name, "GEMM") == 0 );
    _p0 = i;
    _p1 = j;
    _p2 = 0;
     {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[1/*i*/],
                       &new_context,
                       exec_context->function->inout[1/*i*/]->dep_out[0/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[1/*i*/],
                       rank, data);
      }
    }
  }
  new_context.function = exec_context->function->inout[2]->dep_out[0]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[2]->dep_out[0]->dplasma->name, "GEMM") == 0 );
    _p0 = i;
    _p1 = j;
    _p2 = 0;
     {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[2/*i*/],
                       &new_context,
                       exec_context->function->inout[2/*i*/]->dep_out[0/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[2/*i*/],
                       rank, data);
      }
    }
  }
  new_context.function = exec_context->function->inout[3]->dep_out[0]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[3]->dep_out[0]->dplasma->name, "GEMM") == 0 );
    _p0 = i;
    _p1 = j;
    _p2 = 0;
     {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[3/*i*/],
                       &new_context,
                       exec_context->function->inout[3/*i*/]->dep_out[0/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[3/*i*/],
                       rank, data);
      }
    }
  }
  data_repo_entry_set_usage_limit(READVALS_repo, eREADVALS->key, usage);
  if( NULL != ready_list )
    __dplasma_schedule(context, ready_list);
  return ret;
}
static int READVALS_hook(dplasma_execution_unit_t* context, dplasma_execution_context_t *exec_context)
{
  (void)context;
  int i = exec_context->locals[0].value;
  i = exec_context->locals[0].value;
  int j = exec_context->locals[1].value;
  j = exec_context->locals[1].value;
  void *USELESS = NULL;
  data_repo_entry_t *eUSELESS = NULL;
  void *A = NULL;
  data_repo_entry_t *eA = NULL;
  void *B = NULL;
  data_repo_entry_t *eB = NULL;
  void *C = NULL;
  data_repo_entry_t *eC = NULL;
  /* remove warnings in case the variable is not used later */
  (void)i;
  (void)j;
  eUSELESS = data_repo_lookup_entry( STARTUP_repo, STARTUP_hash(0), 0 );
  USELESS = eUSELESS->data[0];

  A = A(i, j);

  B = B(i, j);

  C = C(i, j);


#ifdef HAVE_PAPI
  int i, num_events;
  int events[MAX_EVENTS];
  PAPI_list_events(eventSet, &events, &num_events);
  long long values[num_events];
  PAPI_start(eventSet);
#endif

#if defined(DPLASMA_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, USELESS);
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
  cache_buf_referenced(context->closest_cache, C);
#endif /* DPLASMA_CACHE_AWARENESS */
  TAKE_TIME(context, READVALS_start_key, READVALS_hash(i, j));
  #line 82 "gemm.jdf"
  /* Emptiness */

#line 630 "dgemm.c"

  TAKE_TIME(context, READVALS_end_key, READVALS_hash(i, j));

#ifdef HAVE_PAPI
  PAPI_stop(eventSet, &values);
  if(num_events > 0) {
    printf("PAPI counter values from READVALS (thread=%ld): ", context->eu_id);
    for(i=0; i<num_events; ++i) {
      char event_name[PAPI_MAX_STR_LEN];
      PAPI_event_code_to_name(events[i], &event_name);
      printf("   %s  %lld ", event_name, values[i]);
    }
    printf("\n");
  }
#endif

#if defined(DPLASMA_GRAPHER)
if( NULL != __dplasma_graph_file ) {
  char tmp[128];
  dplasma_service_to_string(exec_context, tmp, 128);
  fprintf(__dplasma_graph_file,
          "%s [shape=\"ellipse\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\",tooltip=\"READVALS%ld\"];\n",
          tmp, colors[context->eu_id], tmp, READVALS_hash(i, j));
}
#endif /* defined(DPLASMA_GRAPHER) */
  {
    void *data[3];
    data[0] = A;
    data[1] = B;
    data[2] = C;
    READVALS_release_dependencies(context, exec_context, 1, data);
  }
  data_repo_entry_used_once( STARTUP_repo, eUSELESS->key );
  return 0;
}

static expr_t expr50 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr52 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr53 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr51 = { .op = 25, .flags = 0, .bop1 = &expr52, .bop2 = &expr53, .value = 0 }; /* SIZE - 1 */
static symbol_t symb10 = { .flags = 0x00000002, .name = "i", .min = &expr50, .max = &expr51 };
static expr_t expr54 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr56 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr57 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr55 = { .op = 25, .flags = 0, .bop1 = &expr56, .bop2 = &expr57, .value = 0 }; /* SIZE - 1 */
static symbol_t symb11 = { .flags = 0x00000002, .name = "j", .min = &expr54, .max = &expr55 };
static expr_t expr58 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 0 }; /* 0 */
static expr_t expr60 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr61 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr59 = { .op = 25, .flags = 0, .bop1 = &expr60, .bop2 = &expr61, .value = 0 }; /* SIZE - 1 */
static symbol_t symb12 = { .flags = 0x00000002, .name = "k", .min = &expr58, .max = &expr59 };
static int inline_expr2( const  assignment_t *assignments )
{
  int i = assignments[0].value;
  return (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
}
static expr_t inline2 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr2 };
static int inline_expr3( const  assignment_t *assignments )
{
  int j = assignments[1].value;
  return (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
}
static expr_t inline3 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr3 };
static int inline_expr4( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==(0);
}
static expr_t inline4 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr4 };
static expr_t expr62 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr63 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep9 = { .cond = &inline4, .dplasma = NULL,
                       .call_params = {&expr62, &expr63, NULL, NULL, NULL}};
static int inline_expr5( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==(0));
}
static expr_t inline5 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr5 };
static expr_t expr64 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr68 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr69 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr67 = { .op = 23, .flags = 0, .bop1 = &expr68, .bop2 = &expr69, .value = 0 }; /* j + SIZE */
static expr_t expr70 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr66 = { .op = 25, .flags = 0, .bop1 = &expr67, .bop2 = &expr70, .value = 0 }; /* j + SIZE - 1 */
static expr_t expr71 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr65 = { .op = 20, .flags = 0, .bop1 = &expr66, .bop2 = &expr71, .value = 0 }; /* j + SIZE - 1 % SIZE */
static expr_t expr73 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr74 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr72 = { .op = 25, .flags = 0, .bop1 = &expr73, .bop2 = &expr74, .value = 0 }; /* k - 1 */
static dep_t dep10 = { .cond = &inline5, .dplasma = NULL,
                       .call_params = {&expr64, &expr65, &expr72, NULL, NULL}};
static int inline_expr6( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==((SIZE)-(1));
}
static expr_t inline6 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr6 };
static expr_t expr75 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr76 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep11 = { .cond = &inline6, .dplasma = NULL,
                       .call_params = {&expr75, &expr76, NULL, NULL, NULL}};
static int inline_expr7( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==((SIZE)-(1)));
}
static expr_t inline7 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr7 };
static expr_t expr77 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr80 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr81 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr79 = { .op = 23, .flags = 0, .bop1 = &expr80, .bop2 = &expr81, .value = 0 }; /* j + 1 */
static expr_t expr82 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr78 = { .op = 20, .flags = 0, .bop1 = &expr79, .bop2 = &expr82, .value = 0 }; /* j + 1 % SIZE */
static expr_t expr84 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr85 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr83 = { .op = 23, .flags = 0, .bop1 = &expr84, .bop2 = &expr85, .value = 0 }; /* k + 1 */
static dep_t dep12 = { .cond = &inline7, .dplasma = NULL,
                       .call_params = {&expr77, &expr78, &expr83, NULL, NULL}};
static param_t param3 = { .name = "A", .sym_type = 3, .param_mask = 0x01,
     .dep_in  = {&dep9, &dep10, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep11, &dep12, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static int inline_expr8( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==(0);
}
static expr_t inline8 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr8 };
static expr_t expr86 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr87 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep13 = { .cond = &inline8, .dplasma = NULL,
                       .call_params = {&expr86, &expr87, NULL, NULL, NULL}};
static int inline_expr9( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==(0));
}
static expr_t inline9 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr9 };
static expr_t expr91 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr92 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr90 = { .op = 23, .flags = 0, .bop1 = &expr91, .bop2 = &expr92, .value = 0 }; /* i + SIZE */
static expr_t expr93 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr89 = { .op = 25, .flags = 0, .bop1 = &expr90, .bop2 = &expr93, .value = 0 }; /* i + SIZE - 1 */
static expr_t expr94 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr88 = { .op = 20, .flags = 0, .bop1 = &expr89, .bop2 = &expr94, .value = 0 }; /* i + SIZE - 1 % SIZE */
static expr_t expr95 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr97 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr98 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr96 = { .op = 25, .flags = 0, .bop1 = &expr97, .bop2 = &expr98, .value = 0 }; /* k - 1 */
static dep_t dep14 = { .cond = &inline9, .dplasma = NULL,
                       .call_params = {&expr88, &expr95, &expr96, NULL, NULL}};
static int inline_expr10( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==((SIZE)-(1));
}
static expr_t inline10 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr10 };
static expr_t expr99 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr100 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep15 = { .cond = &inline10, .dplasma = NULL,
                       .call_params = {&expr99, &expr100, NULL, NULL, NULL}};
static int inline_expr11( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==((SIZE)-(1)));
}
static expr_t inline11 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr11 };
static expr_t expr103 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr104 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr102 = { .op = 23, .flags = 0, .bop1 = &expr103, .bop2 = &expr104, .value = 0 }; /* i + 1 */
static expr_t expr105 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb3 }; /* SIZE */
static expr_t expr101 = { .op = 20, .flags = 0, .bop1 = &expr102, .bop2 = &expr105, .value = 0 }; /* i + 1 % SIZE */
static expr_t expr106 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr108 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr109 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr107 = { .op = 23, .flags = 0, .bop1 = &expr108, .bop2 = &expr109, .value = 0 }; /* k + 1 */
static dep_t dep16 = { .cond = &inline11, .dplasma = NULL,
                       .call_params = {&expr101, &expr106, &expr107, NULL, NULL}};
static param_t param5 = { .name = "B", .sym_type = 3, .param_mask = 0x02,
     .dep_in  = {&dep13, &dep14, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep15, &dep16, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static int inline_expr12( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==(0);
}
static expr_t inline12 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr12 };
static expr_t expr110 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr111 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep17 = { .cond = &inline12, .dplasma = NULL,
                       .call_params = {&expr110, &expr111, NULL, NULL, NULL}};
static int inline_expr13( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==(0));
}
static expr_t inline13 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr13 };
static expr_t expr112 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr113 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr115 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr116 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr114 = { .op = 25, .flags = 0, .bop1 = &expr115, .bop2 = &expr116, .value = 0 }; /* k - 1 */
static dep_t dep18 = { .cond = &inline13, .dplasma = NULL,
                       .call_params = {&expr112, &expr113, &expr114, NULL, NULL}};
static int inline_expr14( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return (k)==((SIZE)-(1));
}
static expr_t inline14 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr14 };
static expr_t expr117 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr118 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static dep_t dep19 = { .cond = &inline14, .dplasma = NULL,
                       .call_params = {&expr117, &expr118, NULL, NULL, NULL}};
static int inline_expr15( const  assignment_t *assignments )
{
  int k = assignments[2].value;
  return !((k)==((SIZE)-(1)));
}
static expr_t inline15 = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr15 };
static expr_t expr119 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb10 }; /* i */
static expr_t expr120 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb11 }; /* j */
static expr_t expr122 = { .op = EXPR_OP_SYMB, .flags = 0, .var = &symb12 }; /* k */
static expr_t expr123 = { .op = EXPR_OP_CONST_INT, .flags = 1, .value = 1 }; /* 1 */
static expr_t expr121 = { .op = 23, .flags = 0, .bop1 = &expr122, .bop2 = &expr123, .value = 0 }; /* k + 1 */
static dep_t dep20 = { .cond = &inline15, .dplasma = NULL,
                       .call_params = {&expr119, &expr120, &expr121, NULL, NULL}};
static param_t param7 = { .name = "C", .sym_type = 3, .param_mask = 0x04,
     .dep_in  = {&dep17, &dep18, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
     .dep_out = {&dep19, &dep20, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} };
static int GEMM_release_dependencies(dplasma_execution_unit_t *context,
                                   const dplasma_execution_context_t *exec_context,
                                   int propagate_remote_dep,
                                   void **data)
{
  int ret = 0;
  data_repo_entry_t *eGEMM;
  int i = exec_context->locals[0].value;
  i = exec_context->locals[0].value;
  int j = exec_context->locals[1].value;
  j = exec_context->locals[1].value;
  int k = exec_context->locals[2].value;
  k = exec_context->locals[2].value;
  eGEMM = data_repo_lookup_entry( GEMM_repo, GEMM_hash(i, j, k), 1 );
  if(data) {
    eGEMM->data[0] = data[0];
    eGEMM->data[1] = data[1];
    eGEMM->data[2] = data[2];
  } else {
    eGEMM->data[0] = NULL;
    eGEMM->data[1] = NULL;
    eGEMM->data[2] = NULL;
  }

  dplasma_execution_context_t*   ready_list = NULL;
  uint32_t usage = 0;
  dplasma_execution_context_t new_context = { .function = NULL, .locals = { {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL},  {.sym = NULL}}};
  /* remove warnings about unused context variable*/
  (void)context;
  /* remove warnings in case the variable is not used later */
  (void)i;
  (void)j;
  (void)k;
#ifdef DISTRIBUTED
  if(propagate_remote_dep) {
    dplasma_remote_dep_reset_forwarded(context);
  }
#else
  (void)propagate_remote_dep;  /* silence a warning */
#endif
  new_context.function = exec_context->function->inout[0]->dep_out[1]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[0]->dep_out[1]->dplasma->name, "GEMM") == 0 );
    _p0 = i;
    _p1 = ((j)+(1))%(SIZE);
    _p2 = (k)+(1);
    if(!((k)==((SIZE)-(1)))) {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[0/*i*/],
                       &new_context,
                       exec_context->function->inout[0/*i*/]->dep_out[1/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[0/*i*/],
                       rank, data);
      }
    }
  }
  new_context.function = exec_context->function->inout[1]->dep_out[1]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[1]->dep_out[1]->dplasma->name, "GEMM") == 0 );
    _p0 = ((i)+(1))%(SIZE);
    _p1 = j;
    _p2 = (k)+(1);
    if(!((k)==((SIZE)-(1)))) {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[1/*i*/],
                       &new_context,
                       exec_context->function->inout[1/*i*/]->dep_out[1/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[1/*i*/],
                       rank, data);
      }
    }
  }
  new_context.function = exec_context->function->inout[2]->dep_out[1]->dplasma; /* GEMM */
  { /** iterate now on the params and dependencies to release OUT dependencies */
    int _p0;
    int _p1;
    int _p2;
    assert( strcmp( exec_context->function->inout[2]->dep_out[1]->dplasma->name, "GEMM") == 0 );
    _p0 = i;
    _p1 = j;
    _p2 = (k)+(1);
    if(!((k)==((SIZE)-(1)))) {
      int i = _p0;
      int j = _p1;
      int k = _p2;
      (void)i;
      (void)j;
      (void)k;
      if( (1) && ((((i)/(stileSIZE))%(GRIDrows))==(rowRANK)) && ((((j)/(stileSIZE))%(GRIDcols))==(colRANK)) ) {
        struct dplasma_dependencies_t** i_placeholder = &(new_context.function->deps);
        struct dplasma_dependencies_t** j_placeholder = &((*i_placeholder)->u.next[_p0 - (*i_placeholder)->min]);
        struct dplasma_dependencies_t** k_placeholder = &((*j_placeholder)->u.next[_p1 - (*j_placeholder)->min]);
        new_context.locals[0].sym = new_context.function->locals[0]; /* task GEMM */
        new_context.locals[0].value = _p0;  /* task GEMM local i */
        new_context.locals[1].sym = new_context.function->locals[1]; /* task GEMM */
        new_context.locals[1].value = _p1;  /* task GEMM local j */
        new_context.locals[2].sym = new_context.function->locals[2]; /* task GEMM */
        new_context.locals[2].value = _p2;  /* task GEMM local k */
        usage++;
        ret += dplasma_release_local_OUT_dependencies(context, exec_context, 
                       exec_context->function->inout[2/*i*/],
                       &new_context,
                       exec_context->function->inout[2/*i*/]->dep_out[1/*j*/]->param,
                       k_placeholder, &ready_list);
      } else if (propagate_remote_dep) {
        int rank, rrank, crank, ncols;
        rrank = ((i)/(stileSIZE))%(GRIDrows);
        crank = ((j)/(stileSIZE))%(GRIDcols);
        ncols = GRIDcols;
        rank = crank + rrank * ncols;
      //DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, crank, rrank, ncols));
        ret += dplasma_remote_dep_activate_rank(context,
                       exec_context,
                       exec_context->function->inout[2/*i*/],
                       rank, data);
      }
    }
  }
  data_repo_entry_set_usage_limit(GEMM_repo, eGEMM->key, usage);
  if( NULL != ready_list )
    __dplasma_schedule(context, ready_list);
  return ret;
}
static int GEMM_hook(dplasma_execution_unit_t* context, dplasma_execution_context_t *exec_context)
{
  (void)context;
  int i = exec_context->locals[0].value;
  i = exec_context->locals[0].value;
  int j = exec_context->locals[1].value;
  j = exec_context->locals[1].value;
  int k = exec_context->locals[2].value;
  k = exec_context->locals[2].value;
  void *A = NULL;
  data_repo_entry_t *eA = NULL;
  void *B = NULL;
  data_repo_entry_t *eB = NULL;
  void *C = NULL;
  data_repo_entry_t *eC = NULL;
  /* remove warnings in case the variable is not used later */
  (void)i;
  (void)j;
  (void)k;
  if((k)==(0)) {
    eA = data_repo_lookup_entry( READVALS_repo, READVALS_hash(i, j), 0 );
    A = eA->data[0];
  }
  if(!((k)==(0))) {
    eA = data_repo_lookup_entry( GEMM_repo, GEMM_hash(i, (((j)+(SIZE))-(1))%(SIZE), (k)-(1)), 0 );
    A = eA->data[0];
  }

  if((k)==(0)) {
    eB = data_repo_lookup_entry( READVALS_repo, READVALS_hash(i, j), 0 );
    B = eB->data[1];
  }
  if(!((k)==(0))) {
    eB = data_repo_lookup_entry( GEMM_repo, GEMM_hash((((i)+(SIZE))-(1))%(SIZE), j, (k)-(1)), 0 );
    B = eB->data[1];
  }

  if((k)==(0)) {
    eC = data_repo_lookup_entry( READVALS_repo, READVALS_hash(i, j), 0 );
    C = eC->data[2];
  }
  if(!((k)==(0))) {
    eC = data_repo_lookup_entry( GEMM_repo, GEMM_hash(i, j, (k)-(1)), 0 );
    C = eC->data[2];
  }


#ifdef HAVE_PAPI
  int i, num_events;
  int events[MAX_EVENTS];
  PAPI_list_events(eventSet, &events, &num_events);
  long long values[num_events];
  PAPI_start(eventSet);
#endif

#if defined(DPLASMA_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
  cache_buf_referenced(context->closest_cache, C);
#endif /* DPLASMA_CACHE_AWARENESS */
  TAKE_TIME(context, GEMM_start_key, GEMM_hash(i, j, k));
  #line 109 "gemm.jdf"
        CORE(
            dgemm, (
                PlasmaNoTrans, PlasmaTrans,
                NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
                NB, /*A.nb,*/
                NB, /*A.nb,*/
                -1.0, A /*A(i, k)*/, NB, /*A.nb,*/
                      B /*B(k, j)*/, NB, /*A.nb,*/
                 1.0, C /*C(i, j)*/, NB /*A.nb*/ )
            );
        OUTPUT((
            "CORE_dgemm( %s, %s, %d, %d, %d, %f, A(%d,%d), %d, B(%d,%d), %d, %f, C(%d,%d), %d)\n",
                "PlasmaNoTrans", "PlasmaTrans",
                NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
                NB, /*A.nb,*/
                NB, /*A.nb,*/
                -1.0, i, k, NB, /*A.nb,*/
                      k, j, NB, /*A.nb,*/
                 1.0, i, j, NB /*A.nb*/ )
            );

#line 1127 "dgemm.c"

  if((k)==((SIZE)-(1))) {
    if( A(i, j) != A) memcpy( A(i, j), A, TILE_SIZE);
}
  if((k)==((SIZE)-(1))) {
    if( B(i, j) != B) memcpy( B(i, j), B, TILE_SIZE);
}
  if((k)==((SIZE)-(1))) {
    if( C(i, j) != C) memcpy( C(i, j), C, TILE_SIZE);
}
  TAKE_TIME(context, GEMM_end_key, GEMM_hash(i, j, k));

#ifdef HAVE_PAPI
  PAPI_stop(eventSet, &values);
  if(num_events > 0) {
    printf("PAPI counter values from  GEMM (thread=%ld): ", context->eu_id);
    for(i=0; i<num_events; ++i) {
      char event_name[PAPI_MAX_STR_LEN];
      PAPI_event_code_to_name(events[i], &event_name);
      printf("   %s  %lld ", event_name, values[i]);
    }
    printf("\n");
  }
#endif

#if defined(DPLASMA_GRAPHER)
if( NULL != __dplasma_graph_file ) {
  char tmp[128];
  dplasma_service_to_string(exec_context, tmp, 128);
  fprintf(__dplasma_graph_file,
          "%s [shape=\"egg\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\",tooltip=\"GEMM%ld\"];\n",
          tmp, colors[context->eu_id], tmp, GEMM_hash(i, j, k));
}
#endif /* defined(DPLASMA_GRAPHER) */
  {
    void *data[3];
    data[0] = A;
    data[1] = B;
    data[2] = C;
    GEMM_release_dependencies(context, exec_context, 1, data);
  }
  if((k)==(0)) {
    data_repo_entry_used_once( READVALS_repo, eA->key );
  }
  if(!((k)==(0))) {
    data_repo_entry_used_once( GEMM_repo, eA->key );
  }
  if((k)==(0)) {
    data_repo_entry_used_once( READVALS_repo, eB->key );
  }
  if(!((k)==(0))) {
    data_repo_entry_used_once( GEMM_repo, eB->key );
  }
  if((k)==(0)) {
    data_repo_entry_used_once( READVALS_repo, eC->key );
  }
  if(!((k)==(0))) {
    data_repo_entry_used_once( GEMM_repo, eC->key );
  }
  return 0;
}

static dplasma_t dplasma_array[6] = {
    {
      .name   = "STARTUP",
      .flags  = 0x01,
      .dependencies_mask = 0x01,
      .nb_locals = 1,
      .locals = {&symb7, NULL, NULL, NULL, NULL},
      .preds = {NULL, NULL, NULL, NULL, NULL},
      .inout= {&param0, NULL, NULL, NULL, NULL},
      .deps = NULL,
      .hook = NULL
    },
    {
      .name   = "A",
      .flags  = 0x00,
      .dependencies_mask = 0x00,
      .nb_locals = 0,
      .locals = {NULL, NULL, NULL, NULL, NULL},
      .preds = {NULL, NULL, NULL, NULL, NULL},
      .inout= {NULL, NULL, NULL, NULL, NULL},
      .deps = NULL,
      .hook = NULL
    },
    {
      .name   = "READVALS",
      .flags  = 0x01,
      .dependencies_mask = 0x0f,
      .nb_locals = 2,
      .locals = {&symb8, &symb9, NULL, NULL, NULL},
      .preds = {&inline0, &inline1, NULL, NULL, NULL},
      .inout= {&param1, &param2, &param4, &param6, NULL},
      .deps = NULL,
      .hook = NULL
    },
    {
      .name   = "GEMM",
      .flags  = 0x02,
      .dependencies_mask = 0x07,
      .nb_locals = 3,
      .locals = {&symb10, &symb11, &symb12, NULL, NULL},
      .preds = {&inline2, &inline3, NULL, NULL, NULL},
      .inout= {&param3, &param5, &param7, NULL, NULL},
      .deps = NULL,
      .hook = NULL
    },
    {
      .name   = "B",
      .flags  = 0x00,
      .dependencies_mask = 0x00,
      .nb_locals = 0,
      .locals = {NULL, NULL, NULL, NULL, NULL},
      .preds = {NULL, NULL, NULL, NULL, NULL},
      .inout= {NULL, NULL, NULL, NULL, NULL},
      .deps = NULL,
      .hook = NULL
    },
    {
      .name   = "C",
      .flags  = 0x00,
      .dependencies_mask = 0x00,
      .nb_locals = 0,
      .locals = {NULL, NULL, NULL, NULL, NULL},
      .preds = {NULL, NULL, NULL, NULL, NULL},
      .inout= {NULL, NULL, NULL, NULL, NULL},
      .deps = NULL,
      .hook = NULL
    }};


static int __dplasma_init(void)
{
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb0)->name);
    if( NULL == symbol ) symbol = &symb0;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &GRIDrows)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb1)->name);
    if( NULL == symbol ) symbol = &symb1;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &GRIDcols)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb2)->name);
    if( NULL == symbol ) symbol = &symb2;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &NB)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb3)->name);
    if( NULL == symbol ) symbol = &symb3;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &SIZE)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb4)->name);
    if( NULL == symbol ) symbol = &symb4;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &colRANK)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb5)->name);
    if( NULL == symbol ) symbol = &symb5;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &rowRANK)) ) {
      return rc;
    }
  }
  {
    int rc;
    symbol_t* symbol = dplasma_search_global_symbol((&symb6)->name);
    if( NULL == symbol ) symbol = &symb6;
    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &stileSIZE)) ) {
      return rc;
    }
  }
  STARTUP_repo = data_repo_create_nothreadsafe(4*4096, 1);
  READVALS_repo = data_repo_create_nothreadsafe(4*4096, 3);
  GEMM_repo = data_repo_create_nothreadsafe(4*4096, 3);
  dep0.dplasma = &dplasma_array[1];
  dep0.param = NULL;
  dep1.dplasma = &dplasma_array[2];
  dep1.param = &param1;
  dep2.dplasma = &dplasma_array[0];
  dep2.param = &param0;
  dep3.dplasma = &dplasma_array[1];
  dep3.param = NULL;
  dep4.dplasma = &dplasma_array[3];
  dep4.param = &param3;
  dep5.dplasma = &dplasma_array[4];
  dep5.param = NULL;
  dep6.dplasma = &dplasma_array[3];
  dep6.param = &param5;
  dep7.dplasma = &dplasma_array[5];
  dep7.param = NULL;
  dep8.dplasma = &dplasma_array[3];
  dep8.param = &param7;
  dep9.dplasma = &dplasma_array[2];
  dep9.param = &param2;
  dep10.dplasma = &dplasma_array[3];
  dep10.param = &param3;
  dep11.dplasma = &dplasma_array[1];
  dep11.param = NULL;
  dep12.dplasma = &dplasma_array[3];
  dep12.param = &param3;
  dep13.dplasma = &dplasma_array[2];
  dep13.param = &param4;
  dep14.dplasma = &dplasma_array[3];
  dep14.param = &param5;
  dep15.dplasma = &dplasma_array[4];
  dep15.param = NULL;
  dep16.dplasma = &dplasma_array[3];
  dep16.param = &param5;
  dep17.dplasma = &dplasma_array[2];
  dep17.param = &param6;
  dep18.dplasma = &dplasma_array[3];
  dep18.param = &param7;
  dep19.dplasma = &dplasma_array[5];
  dep19.param = NULL;
  dep20.dplasma = &dplasma_array[3];
  dep20.param = &param7;

  return 0;
}
int load_dplasma_objects( dplasma_context_t* context )
{
  (void)context;
  dplasma_load_array( dplasma_array, 6 );
  dplasma_load_symbols( dplasma_symbols, 7 );
  return 0;
}

int load_dplasma_hooks( dplasma_context_t* context )
{
  dplasma_t* object;

  (void)context;
  if( 0 != __dplasma_init()) {
     return -1;
  }

  object = (dplasma_t*)dplasma_find("STARTUP");
  object->hook = STARTUP_hook;
  object->release_deps = STARTUP_release_dependencies;

  object = (dplasma_t*)dplasma_find("READVALS");
  object->hook = READVALS_hook;
  object->release_deps = READVALS_release_dependencies;

  object = (dplasma_t*)dplasma_find("GEMM");
  object->hook = GEMM_hook;
  object->release_deps = GEMM_release_dependencies;

#ifdef DPLASMA_PROFILING
  dplasma_profiling_add_dictionary_keyword( "STARTUP", "fill:#E52B50",
                                            &STARTUP_start_key, &STARTUP_end_key);
  dplasma_profiling_add_dictionary_keyword( "A", "fill:#7FFFD4",
                                            &A_start_key, &A_end_key);
  dplasma_profiling_add_dictionary_keyword( "READVALS", "fill:#007FFF",
                                            &READVALS_start_key, &READVALS_end_key);
  dplasma_profiling_add_dictionary_keyword( "GEMM", "fill:#000000",
                                            &GEMM_start_key, &GEMM_end_key);
  dplasma_profiling_add_dictionary_keyword( "B", "fill:#0000FF",
                                            &B_start_key, &B_end_key);
  dplasma_profiling_add_dictionary_keyword( "C", "fill:#0095B6",
                                            &C_start_key, &C_end_key);
#endif /* DPLASMA_PROFILING */

  return 0;
}
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP) \
do { \
  int _vmin = (vMIN); \
  int _vmax = (vMAX); \
  (DEPS) = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) + \
                                           (_vmax - _vmin) * sizeof(dplasma_dependencies_union_t)); \
  /*DEBUG(("Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p\n", */\
  /*       (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP))); */\
  (DEPS)->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL; \
  (DEPS)->symbol = (vSYMBOL); \
  (DEPS)->min = _vmin; \
  (DEPS)->max = _vmax; \
  (DEPS)->prev = (PREVDEP); /* chain them backward */ \
  if( NULL != (PREVDEP) ) {\
    (PREVDEP)->flags = DPLASMA_DEPENDENCIES_FLAG_NEXT | DPLASMA_DEPENDENCIES_FLAG_ALLOCATED;\
  }\
} while (0)\

int enumerate_dplasma_tasks(dplasma_context_t* context)
{
  int nbtasks = 0;
  dplasma_t* function;
  dplasma_dependencies_t *deps;
  /* STARTUP */
  {
    int useless, useless_start, useless_end;
    int useless_min, useless_max;
    dplasma_dependencies_t **useless_deps_location;
    function = (dplasma_t*)dplasma_find( "STARTUP" );
    function->deps = NULL;
    DEBUG(("Prepare dependencies tracking for STARTUP\n"));
    useless_start = 0;
    useless_end = 0;
    useless_min = 0x7fffffff;
    useless_max = -1;
    useless_deps_location = &(function->deps);
    for(useless = useless_start; useless <= useless_end; useless++) {
      if( NULL == *useless_deps_location ) {
        useless_min = useless_start;
        useless_max = useless_end;
        assert( -1 != useless_max );
        ALLOCATE_DEP_TRACKING(deps, useless_min, useless_max, "useless", function->locals[0], *useless_deps_location);
        *useless_deps_location = deps;  /* store the deps in the right location */
      }
      nbtasks++;
    }  /* for useless */
  }
  /* READVALS */
  {
    int i, i_start, i_end;
    int i_min, i_max;
    dplasma_dependencies_t **i_deps_location;
    int j, j_start, j_end;
    int j_min, j_max;
    dplasma_dependencies_t **j_deps_location;
    int pred0;
    int pred1;
    function = (dplasma_t*)dplasma_find( "READVALS" );
    function->deps = NULL;
    DEBUG(("Prepare dependencies tracking for READVALS\n"));
    i_start = 0;
    i_end = (SIZE)-(1);
    i_min = 0x7fffffff;
    i_max = -1;
    i_deps_location = &(function->deps);
    for(i = i_start; i <= i_end; i++) {
      pred0 = (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
      if( !(1 && pred0) ) continue;
      if( NULL == *i_deps_location ) {
        {int _i; for(_i = i_start; _i <= i_end; _i++) {
          int i = _i;
          pred0 = (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
          if( !(1 && pred0) ) continue;
          if( _i < i_min ) i_min = i;
          i_max = i;
        }}
        assert( -1 != i_max );
        ALLOCATE_DEP_TRACKING(deps, i_min, i_max, "i", function->locals[0], *i_deps_location);
        *i_deps_location = deps;  /* store the deps in the right location */
      }
      j_start = 0;
      j_end = (SIZE)-(1);
      j_min = 0x7fffffff;
      j_max = -1;
      for(j = j_start; j <= j_end; j++) {
        pred1 = (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
        if( !(1 && pred1) ) continue;
        j_deps_location = &((*i_deps_location)->u.next[i - (*i_deps_location)->min]);
        if( NULL == *j_deps_location ) {
          {int _j; for(_j = j_start; _j <= j_end; _j++) {
            int j = _j;
            pred1 = (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
            if( !(1 && pred1) ) continue;
            if( _j < j_min ) j_min = j;
            j_max = j;
          }}
          assert( -1 != j_max );
          ALLOCATE_DEP_TRACKING(deps, j_min, j_max, "j", function->locals[1], *i_deps_location);
          *j_deps_location = deps;  /* store the deps in the right location */
        }
        nbtasks++;
      }  /* for j */
    }  /* for i */
  }
  /* GEMM */
  {
    int i, i_start, i_end;
    int i_min, i_max;
    dplasma_dependencies_t **i_deps_location;
    int j, j_start, j_end;
    int j_min, j_max;
    dplasma_dependencies_t **j_deps_location;
    int k, k_start, k_end;
    int k_min, k_max;
    dplasma_dependencies_t **k_deps_location;
    int pred0;
    int pred1;
    function = (dplasma_t*)dplasma_find( "GEMM" );
    function->deps = NULL;
    DEBUG(("Prepare dependencies tracking for GEMM\n"));
    i_start = 0;
    i_end = (SIZE)-(1);
    i_min = 0x7fffffff;
    i_max = -1;
    i_deps_location = &(function->deps);
    for(i = i_start; i <= i_end; i++) {
      pred0 = (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
      if( !(1 && pred0) ) continue;
      if( NULL == *i_deps_location ) {
        {int _i; for(_i = i_start; _i <= i_end; _i++) {
          int i = _i;
          pred0 = (((i)/(stileSIZE))%(GRIDrows))==(rowRANK);
          if( !(1 && pred0) ) continue;
          if( _i < i_min ) i_min = i;
          i_max = i;
        }}
        assert( -1 != i_max );
        ALLOCATE_DEP_TRACKING(deps, i_min, i_max, "i", function->locals[0], *i_deps_location);
        *i_deps_location = deps;  /* store the deps in the right location */
      }
      j_start = 0;
      j_end = (SIZE)-(1);
      j_min = 0x7fffffff;
      j_max = -1;
      for(j = j_start; j <= j_end; j++) {
        pred1 = (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
        if( !(1 && pred1) ) continue;
        j_deps_location = &((*i_deps_location)->u.next[i - (*i_deps_location)->min]);
        if( NULL == *j_deps_location ) {
          {int _j; for(_j = j_start; _j <= j_end; _j++) {
            int j = _j;
            pred1 = (((j)/(stileSIZE))%(GRIDcols))==(colRANK);
            if( !(1 && pred1) ) continue;
            if( _j < j_min ) j_min = j;
            j_max = j;
          }}
          assert( -1 != j_max );
          ALLOCATE_DEP_TRACKING(deps, j_min, j_max, "j", function->locals[1], *i_deps_location);
          *j_deps_location = deps;  /* store the deps in the right location */
        }
        k_start = 0;
        k_end = (SIZE)-(1);
        k_min = 0x7fffffff;
        k_max = -1;
        for(k = k_start; k <= k_end; k++) {
          k_deps_location = &((*j_deps_location)->u.next[j - (*j_deps_location)->min]);
          if( NULL == *k_deps_location ) {
            k_min = k_start;
            k_max = k_end;
            assert( -1 != k_max );
            ALLOCATE_DEP_TRACKING(deps, k_min, k_max, "k", function->locals[2], *j_deps_location);
            *k_deps_location = deps;  /* store the deps in the right location */
          }
          nbtasks++;
        }  /* for k */
      }  /* for j */
    }  /* for i */
  }
  dplasma_register_nb_tasks(context, nbtasks);
  return nbtasks;
}

