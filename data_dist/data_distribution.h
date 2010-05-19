/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_ 
#define _DATA_DISTRIBUTION_H_ 

#include <stdarg.h>
#include <stdint.h>

typedef struct DAGuE_ddesc {
    uint32_t myrank;                  // process rank
    uint32_t cores;                   // number of cores used for computation per node
    uint32_t nodes;                   // number of nodes involved in the computation
    uint32_t (*rank_of)(struct DAGuE_ddesc *mat, ...);
    void *   (*data_of)(struct DAGuE_ddesc *mat, ...);
} DAGuE_ddesc_t;




#endif /* _DATA_DISTRIBUTION_H_ */

/*


#include <stdarg.h>

typedef struct DAGuE_ddesc_blockcyclic2D {
   DAGuE_ddesc_t parent;
   int SIZE;
   int NB;
   int Whatnot;
} DAGuE_ddesc_blockcyclic2D_t;

int rank_of_blockcyclic2D(DAGuE_ddesc_t *mat, ...)
{
  DAGuE_ddesc_blockcyclic2D_t *d = (DAGuE_ddesc_blockcyclic2D_t*)mat;
  int n, m;
  va_list ap;
  va_start(ap, mat);

  n = va_arg(ap, int);
  m = va_arg(ap, int);

  va_end(ap);
}

DAGuE_ddesc_t *DAGuE_ddesc_blockcyclic2D_create(int Size, int Block, int WhatNot)
{
  DAGuE_ddesc_blockcyclic2D_t *res;
  res = (DAGuE_ddesc_blockcyclic2D_t*)calloc(1, sizeof(DAGuE_ddesc_blockcyclic2D_t);
  res->parent.rank_of = rank_of_blockcyclic2D;
  res->parent.data_of = data_of_blockcyclic2D;

  return (DAGuE_ddesc_t *)res;
}
      */
