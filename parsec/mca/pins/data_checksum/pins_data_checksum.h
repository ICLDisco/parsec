#ifndef PINS_DATA_CHECKSUM_H
#define PINS_DATA_CHECKSUM_H
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/runtime.h"

BEGIN_C_DECLS

/**
 * Bits of --mca pins_data_checksum_trace.
 *
 * Verifying a read-only flow across the body is what the module always
 * does; these ask for more than that.
 */
/** Print a fingerprint of every tile each task reads and writes. */
#define PARSEC_PINS_DATA_CHECKSUM_TILES      0x1
/** Verify read-only flows again at task completion, not only at body end. */
#define PARSEC_PINS_DATA_CHECKSUM_COMPLETION 0x2

/**
 * Globally exported variable
 */
/** Set by --mca pins_data_checksum_trace; see the component for what it does. */
PARSEC_DECLSPEC extern int parsec_pins_data_checksum_trace;

PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_data_checksum_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_data_checksum_module;
/* static accessor */
mca_base_component_t * pins_data_checksum_static_component(void);

END_C_DECLS

#endif // PINS_DATA_CHECKSUM_H
