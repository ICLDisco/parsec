/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_TEMPLATE_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_TEMPLATE_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/mca/device/device.h"

BEGIN_C_DECLS

struct parsec_device_template_module_s;
typedef struct parsec_device_template_module_s parsec_device_template_module_t;

PARSEC_OBJ_CLASS_DECLARATION(parsec_device_template_module_t);

extern parsec_device_base_component_t parsec_device_template_component;

PARSEC_DECLSPEC extern int parsec_template_output_stream;

struct parsec_device_template_module_s;
typedef struct parsec_device_template_module_s parsec_device_template_module_t;

struct parsec_device_template_module_s {
    parsec_device_module_t super;
};

/**
 * Create and setup a module for a specific deviceid.
 */
int
parsec_device_template_module_init( int deviceid, parsec_device_module_t** module );
/**
 * Release all resources used for a specific device. Free the device.
 */
int
parsec_device_template_module_fini(parsec_device_module_t* device);

END_C_DECLS

#endif  /* PARSEC_DEVICE_TEMPLATE_H_HAS_BEEN_INCLUDED */
