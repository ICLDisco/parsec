/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * MPI communication engine MCA component declaration.
 *
 * This component keeps the existing funnelled MPI transport behind the new
 * comm framework.  The transport implementation remains in
 * comm_mpi_funnelled.c; this header only exposes the component symbol required
 * by the MCA repository.
 */
#ifndef PARSEC_COMM_MPI_H_HAS_BEEN_INCLUDED
#define PARSEC_COMM_MPI_H_HAS_BEEN_INCLUDED

#include "parsec/mca/comm/comm.h"
#include "parsec/datatype_module.h"

BEGIN_C_DECLS

/**
 * MCA component descriptor for the MPI communication engine.
 */
PARSEC_DECLSPEC extern const parsec_comm_base_component_t parsec_comm_mpi_component;

/**
 * MPI datatype backend installed together with the MPI communication engine.
 */
PARSEC_DECLSPEC extern const parsec_datatype_module_t parsec_comm_mpi_datatype_module;

/**
 * Constructor used by the static MCA component table.
 */
PARSEC_DECLSPEC mca_base_component_t *comm_mpi_static_component(void);

END_C_DECLS

#endif  /* PARSEC_COMM_MPI_H_HAS_BEEN_INCLUDED */
