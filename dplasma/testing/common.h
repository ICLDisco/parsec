/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _TESTSCOMMON_H
#define _TESTSCOMMON_H

/* includes used by the testing_*.c */

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




enum iparam_t {
  IPARAM_RANK,       /* Rank                              */
  IPARAM_NNODES,     /* Number of nodes                   */
  IPARAM_NCORES,     /* Number of cores                   */
  IPARAM_NGPUS,      /* Number of GPUs                    */
  IPARAM_PRIO,       /* Switchpoint for priority DAG      */
  IPARAM_P,          /* Rows in the process grid          */
  IPARAM_Q,          /* Columns in the process grid       */
  IPARAM_M,          /* Number of rows of the matrix      */
  IPARAM_N,          /* Number of columns of the matrix   */
  IPARAM_K,          /* RHS or K                          */
  IPARAM_LDA,        /* Leading dimension of A            */
  IPARAM_LDB,        /* Leading dimension of B            */
  IPARAM_LDC,        /* Leading dimension of C            */
  IPARAM_IB,         /* Inner-blocking size               */
  IPARAM_NB,         /* Number of columns in a tile       */
  IPARAM_MB,         /* Number of rows in a tile          */
  IPARAM_SNB,        /* Number of columns in a super-tile */
  IPARAM_SMB,        /* Number of rows in a super-tile    */
  IPARAM_CHECK,      /* Checking activated or not         */
  IPARAM_VERBOSE,    /* How much noise do we want?        */
  IPARAM_SIZEOF
};

#define SET_IBNBMB_DEFAULTS(iparam, IB, NB, MB) do {
    iparam[IPARAM_IB] = (IB);
    iparam[IPARAM_NB] = (NB);
    iparam[IPARAM_MB] = (MB);
}

#define DECLARE_IPARAM_LOCALS(iparam) \
  int rank  = iparam[IPARAM_RANK];\
  int nodes = iparam[IPARAM_NNODES];\
  int cores = iparam[IPARAM_NCORES];\
  int gpus  = iparam[IPARAM_NGPUS];\
  int prio  = iparam[IPARAM_PRIO];\
  int P     = iparam[IPARAM_P];\
  int Q     = iparam[IPARAM_Q];\
  int M     = iparam[IPARAM_M];\
  int N     = iparam[IPARAM_N];\
  int K     = iparam[IPARAM_K];\
  int LDA   = iparam[IPARAM_LDA];\
  int LDB   = iparam[IPARAM_LDB];\
  int LDC   = iparam[IPARAM_LDC];\
  int MB    = iparam[IPARAM_MB];\
  int NB    = iparam[IPARAM_NB];\
  int SMB   = iparam[IPARAM_SMB];\
  int SNB   = iparam[IPARAM_SNB];\
  int check = iparam[IPARAM_CHECK];\
  int loud  = iparam[IPARAM_VERBOSE];\
  (void)rank;(void)nodes;(void)cores;(void)gpus;(void)prios;(void)P;(void)Q;(void)M;(void)N;(void)K;\
  (void)LDA;(void)LDB;(void)LDC;(void)MB;(void)NB;(void)SMB;(void)SNB;(void)check;(void)loud;

/* Define a double type which not pass through the precision generation process */
typedef double DagDouble_t;
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
#define FLOPS_COUNT(FADD,FMUL,PARAMS) (2. * FADD PARAMS + 6. * FMUL PARAMS)
#else 
#define FLOPS_COUNT(FADD,FMUL,PARAMS) (FADD PARAMS + FMUL PARAMS);
#endif

/*******************************
 * globals values
 *******************************/

#if defined(USE_MPI)
extern MPI_Datatype SYNCHRO;
#endif  /* USE_MPI */

extern const int side[2];
extern const int uplo[2];
extern const int diag[2];
extern const int trans[3];
extern const char *sidestr[2];
extern const char *uplostr[2];
extern const char *diagstr[2];
extern const char *transstr[3];

void print_usage(void);

void iparam_default_facto(int* iparam);
void iparam_default_solve(int* iparam);
void iparam_default_gemm(int* iparam);

dague_context_t *setup_dague(int argc, char* argv[], int *iparam);
void cleanup_dague(dague_context_t* dague);

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
#undef max
#undef min 
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }

#endif /* _TESTSCOMMON_H */
