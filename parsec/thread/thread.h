/**
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef THREAD_H_HAS_BEEN_INCLUDED
#define THREAD_H_HAS_BEEN_INCLUDED

#ifdef ARGOBOTS
#define THREAD_FUNC_TYPE void (*)(void *)
#include "parsec/thread/argobots/argobots_interface.h"
#include "parsec/thread/argobots/comm_scheduler.h"
#else
#define THREAD_FUNC_TYPE void *(*)(void *)
#include "parsec/thread/pthread/pthread_interface.h"
#endif /*ARGOBOTS*/


#define PARSEC_THREAD_GET_NUMBER( dague_context, nb ) do {	  \
		*nb = 0; \
		int vp; \
		for ( vp = 0; vp < (dague_context)->nb_vp; ++vp ) \
			*nb += (dague_context)->virtual_processes[vp]->nb_cores; \
	} while(0)


#endif /*THREAD_H_HAS_BEEN_INCLUDED*/
