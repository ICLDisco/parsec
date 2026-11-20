/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_TASKPOOL_MERGE_H_HAS_BEEN_INCLUDED
#define PARSEC_TASKPOOL_MERGE_H_HAS_BEEN_INCLUDED

/**
 * @defgroup parsec_internal_taskpool_merge Automatic merging of PTG taskpools
 * @ingroup parsec_internal_runtime
 *
 * Taskpools given to the same context run side by side: nothing orders the
 * tasks of one after the tasks of another. Two PTG taskpools that work on the
 * same tile, such as
 *
 *     JDF1:  RW A <- descA(m, n)      JDF2:  RW B <- descA(m, n)
 *                 -> descA(m, n)                  -> descA(m, n)
 *
 * may have JDF2 read descA(m, n) before JDF1 wrote it back.
 *
 * With auto merge, the runtime runs such taskpools as if their DAGs had been
 * merged in the order they were given to the context. For every tile (one
 * parsec_data_t) it keeps, in that order, one block per taskpool that reads or
 * writes the tile through a memory reference (`<- desc(...)` or
 * `-> desc(...)`). A task whose memory references name a tile does not start
 * before the block of its taskpool on that tile is active:
 *  - a block that only reads is active once every older block that writes the
 *    tile is done (readers of different taskpools share the tile);
 *  - a block that writes is active once every older block on the tile is done.
 *
 * A block is done when its taskpool will not touch the tile again. That is
 * counted, not guessed:
 *  - when the taskpool is given to the context, its execution space is
 *    enumerated and every memory reference of every local task is counted on
 *    the block of the tile it names;
 *  - a copy of the tile handed on to a successor, to a remote rank or to the
 *    asynchronous write-back of the communication engine holds the block until
 *    the successor completes or the communication is over.
 *
 * Tasks are held before prepare_input, before any device is chosen, so CPU and
 * accelerator incarnations are ordered alike. Only local accesses matter: a
 * memory reference resolves on the rank that owns the tile, and the tile only
 * leaves that rank as a copy.
 *
 * @{
 */

#include "parsec/runtime.h"

BEGIN_C_DECLS

struct parsec_data_s;
struct parsec_data_copy_s;
struct parsec_merge_tracker_s;
struct parsec_merge_taskpool_s;

/** Set up the auto merge state of a context. */
int  parsec_merge_context_init(parsec_context_t *context);

/** Release the auto merge state of a context. */
void parsec_merge_context_fini(parsec_context_t *context);

/**
 * Called by parsec_context_add_taskpool() before the startup hook. When auto
 * merge applies to tp, counts its accesses and queues them behind the ones of
 * the taskpools added before it. Does nothing otherwise.
 */
int  parsec_merge_taskpool_register(parsec_context_t *context,
                                    parsec_taskpool_t *tp);

/**
 * Called when a merged taskpool terminates. Every local task is complete and
 * every communication over, so whatever its blocks still count is released.
 */
void parsec_merge_taskpool_terminated(parsec_taskpool_t *tp);

/**
 * Decide if a task of a merged taskpool may start.
 *
 * @return 1 if the task may run now, 0 if it was parked. A parked task is
 *         given back to the scheduler once its taskpool may touch the tiles it
 *         names, and goes through this gate again.
 */
int  parsec_merge_task_gate(parsec_execution_stream_t *es, parsec_task_t *task);

/**
 * Complete a task of a merged taskpool: run the task class completion, then
 * release what the task held.
 */
int  parsec_merge_complete_execution(parsec_execution_stream_t *es,
                                     parsec_task_t *task);

/**
 * Hold the tile copy is a copy of, on behalf of tp, while something outside
 * any task of tp still uses the copy (a message, a memcpy).
 *
 * @return the tile held, to give to parsec_merge_unhold(), or NULL when tp
 *         does not track that tile.
 */
struct parsec_data_s *parsec_merge_hold(parsec_taskpool_t *tp,
                                        const struct parsec_data_copy_s *copy);

/** Give back a hold taken by parsec_merge_hold(). */
void parsec_merge_unhold(parsec_taskpool_t *tp, struct parsec_data_s *data);

/** True when tp is ordered by auto merge. */
#define PARSEC_TASKPOOL_IS_MERGED(TP) ((NULL != (TP)) && (NULL != (TP)->merge))

END_C_DECLS

/** @} */

#endif  /* PARSEC_TASKPOOL_MERGE_H_HAS_BEEN_INCLUDED */
