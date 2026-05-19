/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * Communication engine MCA framework.
 *
 * The comm framework selects exactly one transport backend for a PaRSEC
 * context.  The selected backend fills the existing parsec_comm_engine_t
 * interface; the rest of the runtime continues to use parsec_ce and the
 * function table from parsec_comm_engine.h.
 *
 * This framework deliberately exposes only selection and teardown entry points.
 * Callers should not reach back into the selected MCA module after init; all
 * transport operations go through the returned parsec_comm_engine_t.
 */
#ifndef PARSEC_COMM_H_HAS_BEEN_INCLUDED
#define PARSEC_COMM_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/parsec_comm_engine.h"
#include "parsec/mca/mca.h"

BEGIN_C_DECLS

/**
 * Common component header for communication engine components.
 *
 * Component-specific state should live in the component source file or in the
 * parsec_comm_engine_t implementation, not in this base type.
 */
struct parsec_comm_base_component_2_0_0 {
    mca_base_component_2_0_0_t      base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct parsec_comm_base_component_2_0_0 parsec_comm_base_component_2_0_0_t;
typedef struct parsec_comm_base_component_2_0_0 parsec_comm_base_component_t;

/**
 * Initialize a communication engine backend.
 *
 * @param[inout] context PaRSEC context that owns the selected communication
 *                       engine instance.
 *
 * @return A fully initialized parsec_comm_engine_t on success, or NULL if this
 *         module cannot initialize for the provided context.
 */
typedef parsec_comm_engine_t *(*parsec_comm_base_module_init_fn_t)(parsec_context_t *context);

/**
 * Communication module contract.
 *
 * The module has a single responsibility at this layer: build and return the
 * concrete parsec_comm_engine_t used by the runtime.  Backend operations
 * themselves are the function pointers stored in that returned engine.
 */
struct parsec_comm_base_module_1_0_0_t {
    parsec_comm_base_module_init_fn_t init;
};

typedef struct parsec_comm_base_module_1_0_0_t parsec_comm_base_module_1_0_0_t;
typedef struct parsec_comm_base_module_1_0_0_t parsec_comm_base_module_t;

typedef struct parsec_comm_module_s {
    const parsec_comm_base_component_t *component;
    parsec_comm_base_module_t           module;
} parsec_comm_module_t;

/**
 * MCA version tuple for the comm framework.
 */
#define PARSEC_COMM_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
    "comm", 2, 0, 0

/**
 * Select and initialize the active communication engine component.
 *
 * This is internal to the runtime wrapper in parsec_comm_engine.c.  It opens all
 * available comm components, keeps only the selected component open, and calls
 * the selected module's init method.
 */
parsec_comm_engine_t *parsec_comm_engine_component_init(parsec_context_t *context);

/**
 * Close the component selected by parsec_comm_engine_component_init().
 *
 * The parsec_comm_engine_t itself must have already been finalized through its
 * fini function before this call; this function only releases the MCA component
 * lifetime.
 */
int parsec_comm_engine_component_fini(void);

END_C_DECLS

#endif  /* PARSEC_COMM_H_HAS_BEEN_INCLUDED */
