#ifndef stats_h
#define stats_h

/* stats-internal define
 *  DPLASMA_STAT_INCREASE(name, value)
 *  DPLASMA_STAT_DECREASE(name, value)
 * and
 *  void dplasma_stats_dump(char *filename, char *prefix);
 *
 * Call dplasma_stats_dump to dump the stats. prefix is prepended to each
 * line that will look like
 *  prefix: name_of_stat   MAX = %llu
 *
 */

#include "stats-internal.h"

/* Add a DECLARE_STAT line per statitic you want to manage
 * during the execution
 */
DECLARE_STAT(counter_nbtasks);
DECLARE_STAT(mem_bitarray);
DECLARE_STAT(mem_hashtable);
DECLARE_STAT(mem_contexts);
DECLARE_STAT(mem_communications);
DECLARE_STATMAX(counter_hashtable_collisions_size);
DECLARE_STATACC(time_starved);

#endif /* stats_h */
