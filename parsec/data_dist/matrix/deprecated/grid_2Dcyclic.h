/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __GRID_2DCYCLIC_H__
#error "Deprecated headers must not be included directly!"
#endif // __GRID_2DCYCLIC_H__

typedef parsec_grid_2Dcyclic_t grid_2Dcyclic_t __parsec_attribute_deprecated__("Use parsec_grid_2Dcyclic_t");

static inline
void grid_2Dcyclic_init(parsec_grid_2Dcyclic_t* grid, int rank, int P, int Q, int kp, int kq, int ip, int jq)
    __parsec_attribute_deprecated__("Use parsec_grid_2Dcyclic_init");

static inline
void grid_2Dcyclic_init(parsec_grid_2Dcyclic_t* grid, int rank, int P, int Q, int kp, int kq, int ip, int jq)
{
    parsec_grid_2Dcyclic_init(grid, rank, P, Q, kp, kq, ip, jq);
}
