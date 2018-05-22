/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_H_HAS_BEEN_INCLUDED
#define PARSEC_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/datatype.h"
#include "parsec/scheduling.h"
#include "parsec/remote_dep.h"
#include "parsec/datarepo.h"
#include "parsec/data.h"
#include "parsec/mempool.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"

#include "parsec/mca/pins/pins.h"

#include "parsec/interfaces/interface.h"
#include "parsec/interfaces/superscalar/insert_function.h"

#if defined(PARSEC_PROF_GRAPHER)
#include "parsec/parsec_prof_grapher.h"
#endif  /* defined(PARSEC_PROF_GRAPHER) */
#include "parsec/devices/device.h"
#if defined(PARSEC_HAVE_CUDA)
#include "parsec/devices/cuda/dev_cuda.h"
#endif  /* defined(PARSEC_HAVE_CUDA) */

BEGIN_C_DECLS

/**
 * @defgroup parsec_public Header File
 * @ingroup parsec_public
 *   PaRSEC Core routines belonging to the PaRSEC Runtime System.
 *
 *   This is the public API of the PaRSEC runtime system. Functions of
 *   this module are used to manipulate PaRSEC highest level objects.
 *
 *  @{
 */

/**  @} */

END_C_DECLS

#endif  /* PARSEC_H_HAS_BEEN_INCLUDED */
