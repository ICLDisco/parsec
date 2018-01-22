/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _TESTSCOMMON_H
#define _TESTSCOMMON_H

/* includes used by the testing_*.c */
#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* Plasma and math libs */
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <core_blas.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"
#include "parsec/parsec_internal.h"
#include "dplasma.h"
/* timings */
#include "common_timing.h"

#include "flops.h"

/* these are globals in common.c */
extern char *PARSEC_SCHED_NAME[];
extern int unix_timestamp;
extern char cwd[];

/* Update PASTE_CODE_PROGRESS_KERNEL below if you change this list */
enum iparam_t {
  IPARAM_RANK,         /* Rank                              */
  IPARAM_NNODES,       /* Number of nodes                   */
  IPARAM_NCORES,       /* Number of cores                   */
  IPARAM_NGPUS,        /* Number of GPUs                    */
  IPARAM_P,            /* Rows in the process grid          */
  IPARAM_Q,            /* Columns in the process grid       */
  IPARAM_M,            /* Number of rows of the matrix      */
  IPARAM_N,            /* Number of columns of the matrix   */
  IPARAM_K,            /* RHS or K                          */
  IPARAM_LDA,          /* Leading dimension of A            */
  IPARAM_LDB,          /* Leading dimension of B            */
  IPARAM_LDC,          /* Leading dimension of C            */
  IPARAM_IB,           /* Inner-blocking size               */
  IPARAM_NB,           /* Number of columns in a tile       */
  IPARAM_MB,           /* Number of rows in a tile          */
  IPARAM_SNB,          /* Number of columns in a super-tile */
  IPARAM_SMB,          /* Number of rows in a super-tile    */
  IPARAM_HMB,          /* Small MB for recursive hdags */
  IPARAM_HNB,          /* Small NB for recursive hdags */
  IPARAM_CHECK,        /* Checking activated or not         */
  IPARAM_CHECKINV,     /* Inverse Checking activated or not */
  IPARAM_ASYNC,        /* Bench the asynchronous version    */
  IPARAM_VERBOSE,      /* How much noise do we want?        */
  IPARAM_LOWLVL_TREE,  /* Tree used for reduction inside nodes  (specific to xgeqrf_param) */
  IPARAM_HIGHLVL_TREE, /* Tree used for reduction between nodes (specific to xgeqrf_param) */
  IPARAM_QR_TS_SZE,    /* Size of TS domain                     (specific to xgeqrf_param) */
  IPARAM_QR_HLVL_SZE,  /* Size of the high level tree           (specific to xgeqrf_param) */
  IPARAM_RANDOM_SEED,  /* Seed for the pseudo-random generators */
  IPARAM_MATRIX_INIT,  /* Matrix generator type */
  IPARAM_QR_DOMINO,    /* Enable/disable the domino between the upper and the lower tree (specific to xgeqrf_param) */
  IPARAM_QR_TSRR,      /* Enable/disable the round-robin on TS domain */
  IPARAM_BUT_LEVEL,    /* Butterfly level */
  IPARAM_SCHEDULER,    /* User-selected scheduler */
  IPARAM_SIZEOF
};

#define PARSEC_SCHEDULER_DEFAULT 0
#define PARSEC_SCHEDULER_LFQ 1
#define PARSEC_SCHEDULER_LTQ 2
#define PARSEC_SCHEDULER_AP  3
#define PARSEC_SCHEDULER_LHQ 4
#define PARSEC_SCHEDULER_GD  5
#define PARSEC_SCHEDULER_PBQ 6
#define PARSEC_SCHEDULER_IP  7
#define PARSEC_SCHEDULER_RND 8

void iparam_default_facto(int* iparam);
void iparam_default_solve(int* iparam);
void iparam_default_gemm(int* iparam);
void iparam_default_ibnbmb(int* iparam, int ib, int nb, int mb);

#define PASTE_CODE_IPARAM_LOCALS(iparam)                                \
    int rank  = iparam[IPARAM_RANK];                                    \
    int nodes = iparam[IPARAM_NNODES];                                  \
    int cores = iparam[IPARAM_NCORES];                                  \
    int gpus  = iparam[IPARAM_NGPUS];                                   \
    int P     = iparam[IPARAM_P];                                       \
    int Q     = iparam[IPARAM_Q];                                       \
    int M     = iparam[IPARAM_M];                                       \
    int N     = iparam[IPARAM_N];                                       \
    int K     = iparam[IPARAM_K];                                       \
    int NRHS  = K;                                                      \
    int LDA   = max(M, iparam[IPARAM_LDA]);                             \
    int LDB   = max(N, iparam[IPARAM_LDB]);                             \
    int LDC   = max(K, iparam[IPARAM_LDC]);                             \
    int IB    = iparam[IPARAM_IB];                                      \
    int MB    = iparam[IPARAM_MB];                                      \
    int NB    = iparam[IPARAM_NB];                                      \
    int SMB   = iparam[IPARAM_SMB];                                     \
    int SNB   = iparam[IPARAM_SNB];                                     \
    int HMB   = iparam[IPARAM_HMB];                                     \
    int HNB   = iparam[IPARAM_HNB];                                     \
    int MT    = (M%MB==0) ? (M/MB) : (M/MB+1);                          \
    int NT    = (N%NB==0) ? (N/NB) : (N/NB+1);                          \
    int KT    = (K%MB==0) ? (K/MB) : (K/MB+1);                          \
    int check = iparam[IPARAM_CHECK];                                   \
    int check_inv = iparam[IPARAM_CHECKINV];                            \
    int loud  = iparam[IPARAM_VERBOSE];                                 \
    int scheduler = iparam[IPARAM_SCHEDULER];                           \
    int random_seed = iparam[IPARAM_RANDOM_SEED];                       \
    int matrix_init = iparam[IPARAM_MATRIX_INIT];                       \
    int butterfly_level = iparam[IPARAM_BUT_LEVEL];                     \
    int async = iparam[IPARAM_ASYNC];                                   \
    (void)rank;(void)nodes;(void)cores;(void)gpus;(void)P;(void)Q;(void)M;(void)N;(void)K;(void)NRHS; \
    (void)LDA;(void)LDB;(void)LDC;(void)IB;(void)MB;(void)NB;(void)MT;(void)NT;(void)KT; \
    (void)SMB;(void)SNB;(void)HMB;(void)HNB;(void)check;(void)loud;(void)async; \
    (void)scheduler;(void)butterfly_level;(void)check_inv;(void)random_seed;(void)matrix_init;

/* Define a double type which not pass through the precision generation process */
typedef double DagDouble_t;
#define PASTE_CODE_FLOPS( FORMULA, PARAMS ) \
  double gflops = -1.0, flops = FORMULA PARAMS;

#if defined(PRECISION_z) || defined(PRECISION_c)
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops = -1.0, flops = (2. * FADD PARAMS + 6. * FMUL PARAMS);
#else
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops = -1.0, flops = (FADD PARAMS + FMUL PARAMS);
#endif

/*******************************
 * globals values
 *******************************/

#if defined(PARSEC_HAVE_MPI)
extern MPI_Datatype SYNCHRO;
#endif  /* PARSEC_HAVE_MPI */

extern const int side[2];
extern const int uplo[2];
extern const int diag[2];
extern const int trans[3];
extern const int norms[4];
extern const char *sidestr[2];
extern const char *uplostr[2];
extern const char *diagstr[2];
extern const char *transstr[3];
extern const char *normsstr[4];

void print_usage(void);

parsec_context_t *setup_parsec(int argc, char* argv[], int *iparam);
void cleanup_parsec(parsec_context_t* parsec, int *iparam);

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
#undef max
#undef min
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }


/* Paste code to allocate a matrix in desc if cond_init is true */
#define PASTE_CODE_ALLOCATE_MATRIX(DC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DC.mat = parsec_data_allocate((size_t)DC.super.nb_local_tiles * \
                                        (size_t)DC.super.bsiz *      \
                                        (size_t)parsec_datadist_getsizeoftype(DC.super.mtype)); \
        parsec_data_collection_set_key((parsec_data_collection_t*)&DC, #DC);          \
    }

#define PASTE_CODE_ENQUEUE_KERNEL(PARSEC, KERNEL, PARAMS)               \
    SYNC_TIME_START();                                                  \
    parsec_taskpool_t* PARSEC_##KERNEL = dplasma_##KERNEL##_New PARAMS;   \
    PARSEC_CHECK_ERROR(parsec_enqueue(PARSEC, PARSEC_##KERNEL), "parsec_enqueue");      \
    if( loud > 2 ) SYNC_TIME_PRINT(rank, ( #KERNEL "\tDAG created\n"));


#define PASTE_CODE_PROGRESS_KERNEL(PARSEC, KERNEL)                      \
    SYNC_TIME_START();                                                  \
    PARSEC_CHECK_ERROR(parsec_context_start(PARSEC), "parsec_context_start"); \
    TIME_START();                                                       \
    PARSEC_CHECK_ERROR(parsec_context_wait(PARSEC), "parsec_context_wait"); \
    SYNC_TIME_PRINT(rank, (#KERNEL "\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n", \
                           P, Q, NB, N,                                 \
                           gflops=(flops/1e9)/sync_time_elapsed));      \
    PROFILING_SAVE_dINFO("TIME_ELAPSED", time_elapsed);                 \
    PROFILING_SAVE_dINFO("SYNC_TIME_ELAPSED", sync_time_elapsed);       \
    PROFILING_SAVE_dINFO("GFLOPS", gflops);                             \
    PROFILING_SAVE_iINFO("PARAM_RANK", iparam[IPARAM_RANK]);            \
    PROFILING_SAVE_iINFO("PARAM_NNODES", iparam[IPARAM_NNODES]);        \
    PROFILING_SAVE_iINFO("PARAM_NCORES", iparam[IPARAM_NCORES]);        \
    PROFILING_SAVE_iINFO("PARAM_NGPUS", iparam[IPARAM_NGPUS]);          \
    PROFILING_SAVE_iINFO("PARAM_P", iparam[IPARAM_P]);                  \
    PROFILING_SAVE_iINFO("PARAM_Q", iparam[IPARAM_Q]);                  \
    PROFILING_SAVE_iINFO("PARAM_M", iparam[IPARAM_M]);                  \
    PROFILING_SAVE_iINFO("PARAM_N", iparam[IPARAM_N]);                  \
    PROFILING_SAVE_iINFO("PARAM_K", iparam[IPARAM_K]);                  \
    PROFILING_SAVE_iINFO("PARAM_LDA", iparam[IPARAM_LDA]);              \
    PROFILING_SAVE_iINFO("PARAM_LDB", iparam[IPARAM_LDB]);              \
    PROFILING_SAVE_iINFO("PARAM_LDC", iparam[IPARAM_LDC]);              \
    PROFILING_SAVE_iINFO("PARAM_IB", iparam[IPARAM_IB]);                \
    PROFILING_SAVE_iINFO("PARAM_NB", iparam[IPARAM_NB]);                \
    PROFILING_SAVE_iINFO("PARAM_MB", iparam[IPARAM_MB]);                \
    PROFILING_SAVE_iINFO("PARAM_SNB", iparam[IPARAM_SNB]);              \
    PROFILING_SAVE_iINFO("PARAM_SMB", iparam[IPARAM_SMB]);              \
    PROFILING_SAVE_iINFO("PARAM_HNB", iparam[IPARAM_HNB]);              \
    PROFILING_SAVE_iINFO("PARAM_CHECK", iparam[IPARAM_CHECK]);          \
    PROFILING_SAVE_iINFO("PARAM_CHECKINV", iparam[IPARAM_CHECKINV]);    \
    PROFILING_SAVE_iINFO("PARAM_VERBOSE", iparam[IPARAM_VERBOSE]);      \
    PROFILING_SAVE_iINFO("PARAM_LOWLVL_TREE", iparam[IPARAM_LOWLVL_TREE]); \
    PROFILING_SAVE_iINFO("PARAM_HIGHLVL_TREE", iparam[IPARAM_HIGHLVL_TREE]); \
    PROFILING_SAVE_iINFO("PARAM_QR_TS_SZE", iparam[IPARAM_QR_TS_SZE]);  \
    PROFILING_SAVE_iINFO("PARAM_QR_HLVL_SZE", iparam[IPARAM_QR_HLVL_SZE]); \
    PROFILING_SAVE_iINFO("PARAM_QR_DOMINO", iparam[IPARAM_QR_DOMINO]);  \
    PROFILING_SAVE_iINFO("PARAM_QR_TSRR", iparam[IPARAM_QR_TSRR]);      \
    PROFILING_SAVE_iINFO("PARAM_BUT_LEVEL", iparam[IPARAM_BUT_LEVEL]);  \
    PROFILING_SAVE_iINFO("PARAM_SCHEDULER", iparam[IPARAM_SCHEDULER]);  \
    if(loud >= 5 && rank == 0) {                                        \
        printf("<DartMeasurement name=\"performance\" type=\"numeric/double\"\n" \
               "                 encoding=\"none\" compression=\"none\">\n" \
               "%g\n"                                                   \
               "</DartMeasurement>\n",                                  \
               gflops);                                                 \
    }                                                                   \
    (void)gflops;


#endif /* _TESTSCOMMON_H */
