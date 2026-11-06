/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * Backend datatype module interface.
 *
 * PaRSEC keeps the public parsec_type_* API stable, but the implementation of
 * those routines is provided by the communication backend selected for the
 * process.  This lets an MPI backend keep using MPI datatypes while another
 * backend can provide a different representation.
 *
 * Datatype matching is intentionally not part of this module.  The current
 * parsec_type_match() API is a lightweight compatibility helper and does not
 * require backend-specific layout comparison.
 */
#ifndef PARSEC_DATATYPE_MODULE_H_HAS_BEEN_INCLUDED
#define PARSEC_DATATYPE_MODULE_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/datatype.h"

BEGIN_C_DECLS

typedef int (*parsec_datatype_module_size_fn_t)(parsec_datatype_t type,
                                                int *size);
typedef int (*parsec_datatype_module_extent_fn_t)(parsec_datatype_t type,
                                                  ptrdiff_t *lb,
                                                  ptrdiff_t *extent);
typedef int (*parsec_datatype_module_free_fn_t)(parsec_datatype_t *type);
typedef int (*parsec_datatype_module_create_contiguous_fn_t)(int count,
                                                             parsec_datatype_t oldtype,
                                                             parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_vector_fn_t)(int count,
                                                         int blocklength,
                                                         int stride,
                                                         parsec_datatype_t oldtype,
                                                         parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_hvector_fn_t)(int count,
                                                          int blocklength,
                                                          ptrdiff_t stride,
                                                          parsec_datatype_t oldtype,
                                                          parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_indexed_fn_t)(int count,
                                                          const int array_of_blocklengths[],
                                                          const int array_of_displacements[],
                                                          parsec_datatype_t oldtype,
                                                          parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_indexed_block_fn_t)(int count,
                                                                int blocklength,
                                                                const int array_of_displacements[],
                                                                parsec_datatype_t oldtype,
                                                                parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_struct_fn_t)(int count,
                                                         const int array_of_blocklengths[],
                                                         const ptrdiff_t array_of_displacements[],
                                                         const parsec_datatype_t array_of_types[],
                                                         parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_create_resized_fn_t)(parsec_datatype_t oldtype,
                                                          ptrdiff_t lb,
                                                          ptrdiff_t extent,
                                                          parsec_datatype_t *newtype);
typedef int (*parsec_datatype_module_contiguous_fn_t)(parsec_datatype_t type);

typedef struct parsec_datatype_module_s {
    parsec_datatype_module_size_fn_t                 size;
    parsec_datatype_module_extent_fn_t               extent;
    parsec_datatype_module_free_fn_t                 free;
    parsec_datatype_module_create_contiguous_fn_t    create_contiguous;
    parsec_datatype_module_create_vector_fn_t        create_vector;
    parsec_datatype_module_create_hvector_fn_t       create_hvector;
    parsec_datatype_module_create_indexed_fn_t       create_indexed;
    parsec_datatype_module_create_indexed_block_fn_t create_indexed_block;
    parsec_datatype_module_create_struct_fn_t        create_struct;
    parsec_datatype_module_create_resized_fn_t       create_resized;
    parsec_datatype_module_contiguous_fn_t           contiguous;
} parsec_datatype_module_t;

/**
 * Install the datatype backend used by the public parsec_type_* API.
 *
 * The selected communication component calls this during initialization.  The
 * installed module must remain valid until the communication engine is
 * finalized, at which point parsec_datatype_module_uninstall() removes it.
 */
int parsec_datatype_module_install(const parsec_datatype_module_t *module);

/**
 * Remove the datatype backend installed by the selected communication
 * component, so that no parsec_type_* call can reach the transport once it is
 * finalized. They then fail with PARSEC_ERR_NOT_FOUND, unless the build has a
 * transport-independent fallback.
 */
int parsec_datatype_module_uninstall(void);

END_C_DECLS

#endif  /* PARSEC_DATATYPE_MODULE_H_HAS_BEEN_INCLUDED */
