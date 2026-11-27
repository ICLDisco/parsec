/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "parsec/parsec_config.h"
#include "parsec/mca/comm/comm.h"
#include "parsec/mca/device/device.h"
#include "parsec/remote_dep.h"
#include "parsec/utils/mca_param.h"

parsec_comm_engine_t parsec_ce;

uint8_t parsec_comm_buffer_device_types = PARSEC_DEV_CPU;

/**
 * @brief Honor a setting of the removed mca mpi_gpu_aware.
 *
 * The parameter used to be an integer: 0 forbade device memory in both
 * directions, and any other value picked send, receive or both. The split is
 * gone. 0 is the one value that still has a meaning, and it is ^gpu. Anything
 * else is reported and ignored, so a script that still sets the old name does
 * not silently keep handing device pointers to the engine, or silently stop
 * forbidding them.
 */
static void parsec_comm_engine_honor_legacy_gpu_aware(uint8_t *disabled)
{
    int legacy_value = -1;
    parsec_mca_param_source_t source = MCA_PARAM_SOURCE_DEFAULT;
    int index = parsec_mca_param_reg_int_name("mpi", "gpu_aware",
                                              "Replaced by runtime_comm_use_device_types. 0 turns every accelerator"
                                              " off, as runtime_comm_use_device_types=^gpu does. Any other value is"
                                              " ignored: send and receive are no longer separate, and the engine uses"
                                              " what the backend reports.",
                                              false, false, -1, &legacy_value);

    if( (index < 0) ||
        (PARSEC_SUCCESS != parsec_mca_param_lookup_source(index, &source, NULL)) ||
        (MCA_PARAM_SOURCE_DEFAULT == source) ) {
        return;
    }
    if( 0 == legacy_value ) {
        *disabled |= PARSEC_DEV_GPU_MASK;
        parsec_warning("The mca mpi_gpu_aware=0 is replaced by runtime_comm_use_device_types=^gpu."
                       " No accelerator memory will be handed to the communication engine.");
        return;
    }
    parsec_warning("The mca mpi_gpu_aware=%d is ignored. It is replaced by runtime_comm_use_device_types,"
                   " and send and receive are no longer chosen separately. The engine uses what the backend"
                   " reports, minus any type runtime_comm_use_device_types turns off.",
                   legacy_value);
}

/**
 * @brief Read which accelerators the user wants used, and which turned off.
 *
 * A name on its own asks for that accelerator and a name behind a '^' turns it
 * off, the same spelling other mca parameters use. The two are not mirror
 * images, because the answer can never be more than the backend is capable of:
 * asking for an accelerator cannot conjure up support for it, so asking is a
 * statement of expectation, worth a word when it is not met, while '^' is the
 * only half that moves anything.
 *
 * Only accelerators are in question. Host memory is not something a backend
 * can decline, and the types that stand for no memory at all are not something
 * to hand anyone, so a name outside the accelerators is refused rather than
 * quietly folded in. A name nobody recognises costs that name and no more:
 * dropping the rest of the list would turn one typo into a silent change of
 * everything else.
 *
 * @param[out] demanded
 *          The accelerators asked for by name. "gpu" asks for whatever is
 *          there rather than for anything in particular, so it is not a demand
 *          that anyone can fall short of.
 *
 * @param[out] disabled
 *          The accelerators to keep away from the backend, able or not.
 */
static void parsec_comm_engine_device_types_to_use(uint8_t *demanded, uint8_t *disabled)
{
    char *spec = NULL;

    *demanded = *disabled = PARSEC_DEV_NONE;
    parsec_comm_engine_honor_legacy_gpu_aware(disabled);

    parsec_mca_param_reg_string_name("runtime", "comm_use_device_types",
                                     "Comma-separated accelerator types the communication engine may be handed"
                                     " directly: cuda, hip (or rocm), level_zero. A bare name is an expectation."
                                     " It is reported when the backend cannot take that type, and it never enables"
                                     " a type the backend did not report. A name behind a '^', as in ^cuda, turns"
                                     " that type off whether or not the backend can take it. gpu and all name more"
                                     " than one type, so neither is an expectation; ^gpu turns every accelerator off."
                                     " Memory of a type left out takes the detour through host memory, where it"
                                     " always has a copy. Unset by default, which uses whatever the backend reports.",
                                     false, false, "", &spec);
    if( NULL == spec ) return;

    for( const char *token = spec; '\0' != *token; ) {
        const char *comma = strchr(token, ',');
        const char *token_end = (NULL == comma) ? token + strlen(token) : comma;
        int token_length = (int)(token_end - token);
        int turn_off = ('^' == *token);
        uint8_t types = parsec_device_type_from_name(token + turn_off, (size_t)(token_length - turn_off));
        uint8_t accelerators = types & PARSEC_DEV_GPU_MASK;

        if( PARSEC_DEV_NONE == types ) {
            parsec_warning("The mca runtime_comm_use_device_types names \"%.*s\", which is not a device type."
                           " It is skipped, and the rest of the list still applies.",
                           token_length, token);
        } else if( PARSEC_DEV_NONE == accelerators ) {
            parsec_warning("The mca runtime_comm_use_device_types names \"%.*s\", which is not an accelerator."
                           " Only accelerators are in question here, host memory being something every"
                           " backend can take. It is skipped, and the rest of the list still applies.",
                           token_length, token);
        } else if( turn_off ) {
            *disabled |= accelerators;
        } else if( 0 == (accelerators & (accelerators - 1)) ) {
            *demanded |= accelerators;  /* one bit is one accelerator asked for by name */
        }
        token = ('\0' == *token_end) ? token_end : token_end + 1;
    }
    free(spec);
}

/**
 * @brief Settle which device types may be handed to the backend.
 *
 * The backend has said what it can do; all that is left is to take out what
 * the user turned off, and to say so where what they asked for and what the
 * backend can do do not meet. Host memory survives either way. This waits
 * until the backend is up because for some of them the question can only be
 * asked of a running library.
 */
static void parsec_comm_engine_publish_buffer_device_types(void)
{
    uint8_t available = parsec_ce.capabilites.buffer_device_types | PARSEC_DEV_CPU;
    uint8_t demanded, disabled;
    char names[MAX_DEVICE_TYPE_NAMES_LEN];

    parsec_comm_engine_device_types_to_use(&demanded, &disabled);
    if( PARSEC_DEV_NONE != (demanded & ~available) ) {
        parsec_device_type_to_names(demanded & ~available, names, sizeof(names));
        parsec_warning("The mca runtime_comm_use_device_types asks for memory of type %s to be handed to the"
                       " communication engine directly, which the selected backend reports it cannot take."
                       " That memory takes the detour through the host instead.", names);
    }
    if( PARSEC_DEV_NONE != (demanded & disabled) ) {
        parsec_device_type_to_names(demanded & disabled, names, sizeof(names));
        parsec_warning("The mca runtime_comm_use_device_types both asks for and turns off memory of type %s."
                       " It stays off, that being the half of the two that can be honoured safely.", names);
    }
    parsec_comm_buffer_device_types = (available & ~disabled) | PARSEC_DEV_CPU;
}

#if defined(PARSEC_HAVE_MPI)

/* Select and initialize the distributed communication backend. */
parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    parsec_comm_engine_t *ce = parsec_comm_engine_component_init(parsec_context);

    /* No component could be selected (the user may have disabled them all).
     * The runtime then runs without any communication capability. */
    if( NULL == ce ) {
        return NULL;
    }
    assert(ce->capabilites.sided > 0 && ce->capabilites.sided < 3);
    parsec_comm_engine_publish_buffer_device_types();
    return ce;
}

extern int remote_dep_ce_fini(parsec_context_t* context);

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    /* A backend fills in the function table as the last step of its
     * initialization, so an empty table means there is nothing to tear down. */
    if( NULL == comm_engine->fini ) {
        return PARSEC_SUCCESS;
    }
    (void) parsec_remote_dep_fini(comm_engine->parsec_context);
    remote_dep_ce_fini(comm_engine->parsec_context);
    /* Finalize the backend engine before releasing the selected MCA component. */
    comm_engine->fini(comm_engine);
    parsec_comm_engine_component_fini();
    return PARSEC_SUCCESS;
}

#else

parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    /* Local builds keep the in-process engine and do not select a comm component. */
    parsec_ce.parsec_context = parsec_context;
    parsec_ce.capabilites.sided = 0;
    parsec_ce.capabilites.supports_noncontiguous_datatype = 0;
    parsec_ce.capabilites.multithreaded = 0;
    /* There is no backend to hand a buffer to, and no remote peer to hand it
     * for, so the question only has its most restrictive answer. */
    parsec_ce.capabilites.buffer_device_types = PARSEC_DEV_CPU;
    parsec_ce.taskpool_sync_ids = NULL;
    parsec_comm_engine_publish_buffer_device_types();
    return &parsec_ce;
}

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    (void)comm_engine;
    return PARSEC_SUCCESS;
}

#endif  /* defined(PARSEC_HAVE_MPI) */
