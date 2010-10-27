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

/*******************************
 * globals values
 *******************************/
/* Define a double type which not pass through the precision generation process */
typedef double DagDouble_t;

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
