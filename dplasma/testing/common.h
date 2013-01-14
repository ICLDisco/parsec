/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _TESTSCOMMON_H
#define _TESTSCOMMON_H

/* includes used by the testing_*.c */
#include "dague_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* Plasma and math libs */
#include <math.h>
#include <cblas.h>
#include <plasma.h>
#include <lapacke.h>
#include <core_blas.h>
/* dague things */
#include "dague.h"
#include "scheduling.h"
#include "profiling.h"
#include "dplasma.h"
/* timings */
#include "common_timing.h"
#ifdef DAGUE_VTRACE
#include "vt_user.h"
#endif

#include "flops.h"

enum iparam_t {
  IPARAM_RANK,         /* Rank                              */
  IPARAM_NNODES,       /* Number of nodes                   */
  IPARAM_NCORES,       /* Number of cores                   */
  IPARAM_SCHEDULER,    /* What scheduler do we choose */
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
  IPARAM_CHECK,        /* Checking activated or not         */
  IPARAM_CHECKINV,     /* Inverse Checking activated or not */
  IPARAM_VERBOSE,      /* How much noise do we want?        */
  IPARAM_LOWLVL_TREE,  /* Tree used for reduction inside nodes  (specific to xgeqrf_param) */
  IPARAM_HIGHLVL_TREE, /* Tree used for reduction between nodes (specific to xgeqrf_param) */
  IPARAM_QR_TS_SZE,    /* Size of TS domain                     (specific to xgeqrf_param) */
  IPARAM_QR_HLVL_SZE,  /* Size of the high level tree           (specific to xgeqrf_param) */
  IPARAM_QR_DOMINO,    /* Enable/disable the domino between the upper and the lower tree (specific to xgeqrf_param) */
  IPARAM_QR_TSRR,      /* Enable/disable the round-robin on TS domain */
  IPARAM_BUT_LEVEL,    /* Butterfly level */
  IPARAM_DOT,          /* Do we require to output the DOT file? */
  IPARAM_SIZEOF
};

void iparam_default_facto(int* iparam);
void iparam_default_solve(int* iparam);
void iparam_default_gemm(int* iparam);
void iparam_default_ibnbmb(int* iparam, int ib, int nb, int mb);

#define PASTE_CODE_IPARAM_LOCALS(iparam) \
  int rank  = iparam[IPARAM_RANK];\
  int nodes = iparam[IPARAM_NNODES];\
  int cores = iparam[IPARAM_NCORES];\
  int gpus  = iparam[IPARAM_NGPUS];\
  int P     = iparam[IPARAM_P];\
  int Q     = iparam[IPARAM_Q];\
  int M     = iparam[IPARAM_M];\
  int N     = iparam[IPARAM_N];\
  int K     = iparam[IPARAM_K];\
  int NRHS  = K;\
  int LDA   = max(M, iparam[IPARAM_LDA]);\
  int LDB   = max(N, iparam[IPARAM_LDB]);\
  int LDC   = max(K, iparam[IPARAM_LDC]);\
  int IB    = iparam[IPARAM_IB];\
  int MB    = iparam[IPARAM_MB];\
  int NB    = iparam[IPARAM_NB];\
  int SMB   = iparam[IPARAM_SMB];\
  int SNB   = iparam[IPARAM_SNB];\
  int MT    = (M%MB==0) ? (M/MB) : (M/MB+1); \
  int NT    = (N%NB==0) ? (N/NB) : (N/NB+1); \
  int KT    = (K%MB==0) ? (K/MB) : (K/MB+1); \
  int check = iparam[IPARAM_CHECK];\
  int check_inv = iparam[IPARAM_CHECKINV];\
  int loud  = iparam[IPARAM_VERBOSE];\
  int scheduler = iparam[IPARAM_SCHEDULER];\
  int nb_local_tasks = 0;                                               \
  int butterfly_level = iparam[IPARAM_BUT_LEVEL];\
  (void)rank;(void)nodes;(void)cores;(void)gpus;(void)P;(void)Q;(void)M;(void)N;(void)K;(void)NRHS; \
  (void)LDA;(void)LDB;(void)LDC;(void)IB;(void)MB;(void)NB;(void)MT;(void)NT;(void)KT;(void)SMB;(void)SNB;(void)check;(void)loud;\
  (void)scheduler;(void)nb_local_tasks; (void)butterfly_level;(void)check_inv;

/* Define a double type which not pass through the precision generation process */
typedef double DagDouble_t;
#define PASTE_CODE_FLOPS( FORMULA, PARAMS ) \
  double gflops, flops = FORMULA PARAMS;

#if defined(PRECISION_z) || defined(PRECISION_c)
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops, flops = (2. * FADD PARAMS + 6. * FMUL PARAMS);
#else
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops, flops = (FADD PARAMS + FMUL PARAMS);
#endif

/*******************************
 * globals values
 *******************************/

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

dague_context_t *setup_dague(int argc, char* argv[], int *iparam);
void cleanup_dague(dague_context_t* dague, int *iparam);

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
#undef max
#undef min
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }


/* Paste code to allocate a matrix in desc if cond_init is true */
#define PASTE_CODE_ALLOCATE_MATRIX(DDESC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DDESC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DDESC.mat = dague_data_allocate((size_t)DDESC.super.nb_local_tiles * \
                                        (size_t)DDESC.super.bsiz *      \
                                        (size_t)dague_datadist_getsizeoftype(DDESC.super.mtype)); \
        dague_ddesc_set_key((dague_ddesc_t*)&DDESC, #DDESC);            \
    }

#define PASTE_CODE_ENQUEUE_KERNEL(DAGUE, KERNEL, PARAMS)                \
    SYNC_TIME_START();                                                  \
    dague_object_t* DAGUE_##KERNEL = dplasma_##KERNEL##_New PARAMS;     \
    dague_enqueue(DAGUE, DAGUE_##KERNEL);                               \
    nb_local_tasks = DAGUE_##KERNEL->nb_local_tasks;                    \
    if(loud) SYNC_TIME_PRINT(rank, ( #KERNEL " DAG creation: %d local tasks enqueued\n", nb_local_tasks));


#define PASTE_CODE_PROGRESS_KERNEL(DAGUE, KERNEL)                       \
    SYNC_TIME_START();                                                  \
    TIME_START();                                                       \
    dague_progress(DAGUE);                                              \
    if( loud > 2 )                                                      \
        TIME_PRINT(rank, (#KERNEL " computed %d tasks,\trate %f task/s\n",    \
                          nb_local_tasks,                               \
                          nb_local_tasks/time_elapsed));                \
    SYNC_TIME_PRINT(rank, (#KERNEL " computation PxQ= %d %d N= %d NB= %d : %f gflops\n", \
                           P, Q, N, NB,                                 \
                           gflops=(flops/1e9)/sync_time_elapsed));      \
    if(loud >= 5 && rank == 0) {                                        \
        printf("<DartMeasurement name=\"performance\" type=\"numeric/double\"\n" \
               "                 encoding=\"none\" compression=\"none\">\n" \
               "%g\n"                                                   \
               "</DartMeasurement>\n",                                  \
               gflops);                                                 \
    }                                                                   \
    (void)gflops;

#endif /* _TESTSCOMMON_H */
