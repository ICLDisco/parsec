/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_H_HAS_BEEN_INCLUDED
#define PARSEC_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

BEGIN_C_DECLS

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
#include "parsec/mca/device/device.h"

END_C_DECLS

#endif  /* PARSEC_H_HAS_BEEN_INCLUDED */
