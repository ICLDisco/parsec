/**
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <string.h>
#include <assert.h>
#include <errno.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

#include "parsec/parsec_hwloc.h"
#include "parsec/vpmap.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"

/* If HWLOC is not available support up to 64 cores */
#define DEFAULT_NB_CORE 64
/**
 * These structures are used by the from_hardware and from_file
 * to store the whole vp map
 */
typedef struct {
    int nbcores;
    int cores[1];
    int ht[1];
} vpmap_thread_t;

typedef struct {
    int nbthreads;
    vpmap_thread_t **threads;
} vpmap_t;

static int vpmap_nb_total_threads = -1; /**< maintains the total number of threads */

static vpmap_t *map = NULL;     /**< used by the from_file and from_affinity only */
static int nbvp = -1;
static int nbht = 1;
static int nbthreadspervp = -1; /**< used by the from_parameters only */
static int nbcores = -1;        /**< used by the from_parameters only */

static int vpmap_get_nb_threads_in_vp_parameters(int vp);
static int vpmap_get_nb_cores_affinity_parameters(int vp, int thread);
static void vpmap_get_core_affinity_parameters(int vp, int thread, int *cores, int *ht);

static int parse_binding_parameter(int vp, int nbth, char * binding);

int vpmap_get_nb_total_threads(void)
{
    return vpmap_nb_total_threads;
}

/* The parameters variant is used while no init has been called,
 * as they return -1 in this case for any call
 */
vpmap_get_nb_threads_in_vp_t vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
vpmap_get_nb_cores_affinity_t vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
vpmap_get_core_affinity_t vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;


int vpmap_get_nb_vp(void)
{
    return nbvp;
}

static int vpmap_get_nb_threads_in_vp_parameters(int vp)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (nbcores == -1) )
        return -1;
    return nbthreadspervp;
}

static int vpmap_get_nb_cores_affinity_parameters(int vp, int thread)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (thread < 0) ||
        (thread >= nbthreadspervp )||
        (nbcores == -1) )
        return -1;
    return 1;
}

static void vpmap_get_core_affinity_parameters(int vp, int thread, int *cores, int *ht)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (thread < 0) ||
        (thread >= nbthreadspervp )||
        (nbcores == -1) )
        return;
#if defined(PARSEC_HAVE_HWLOC)
    //nb_real_cores = parsec_hwloc_nb_real_cores();
    nbht = parsec_hwloc_get_ht();
#endif /* PARSEC_HAVE_HWLOC */
    *cores = (vp * nbcores * nbht) + thread;
    if (nbht > 1 ) {
        *ht = (*cores) % nbht;
    } else {
        *ht = -1;
    }
}

void vpmap_fini(void)
{
    int v, t;
    if( NULL != map ) {
        for(v = 0; v < nbvp; v++) {
            for(t = 0; t < map[v].nbthreads; t++) {
                free(map[v].threads[t]);
                map[v].threads[t] = NULL;
            }
        }
        free(map);
        map = NULL;
    }
    nbvp = -1;
    nbthreadspervp = -1;
    nbcores = -1;
}

static int vpmap_get_nb_threads_in_vp_datamap(int vp)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (map == NULL) )
        return -1;
    return map[vp].nbthreads;
}

static int vpmap_get_nb_cores_affinity_datamap(int vp, int thread)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (map == NULL) ||
        (thread < 0) ||
        (thread >= map[vp].nbthreads ) )
        return -1;
    return map[vp].threads[thread]->nbcores;
}

static void vpmap_get_core_affinity_datamap(int vp, int thread, int *cores, int *ht)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (map == NULL) ||
        (thread < 0) ||
        (thread >= map[vp].nbthreads ) )
        return;
    memcpy(cores, map[vp].threads[thread]->cores, map[vp].threads[thread]->nbcores * sizeof(int));
    memcpy(ht, map[vp].threads[thread]->ht, map[vp].threads[thread]->nbcores * sizeof(int));
}

int vpmap_init_from_hardware_affinity(int nbcores)
{
#if defined(PARSEC_HAVE_HWLOC)
    int vp_id, th_id, core_id, ht_id;

    /* Compute the number of VP according to the number of objects at the
     * lowest level between sockets and NUMA nodes */
    int level = parsec_hwloc_core_first_hrwd_ancestor_depth();
    nbvp = parsec_hwloc_get_nb_objects(level);
    nbht = parsec_hwloc_get_ht();

    if (nbvp <= 0 ) {
        vpmap_init_from_flat(nbcores);
        return 0;
    }

    map = (vpmap_t*)calloc(nbvp, sizeof(vpmap_t));
    /* Define the VP map:
     * threads are distributed in order on the cores (hwloc numbering, ensure locality)
     */
    core_id = 0;
    vpmap_nb_total_threads = 0;

    for( vp_id = 0; vp_id < nbvp; vp_id++ ) {
        nbthreadspervp = parsec_hwloc_nb_cores_per_obj(level, vp_id) * nbht;
        vpmap_nb_total_threads += nbthreadspervp;

        map[vp_id].nbthreads = nbthreadspervp;
        map[vp_id].threads   = (vpmap_thread_t**)calloc(nbthreadspervp, sizeof(vpmap_thread_t*));

        for( th_id = 0; th_id < nbthreadspervp; th_id += nbht ) {
            for( ht_id = 0; ht_id < nbht ; ht_id++ ) {
                map[vp_id].threads[th_id + ht_id] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                map[vp_id].threads[th_id + ht_id]->nbcores = 1;
                map[vp_id].threads[th_id + ht_id]->cores[0] = core_id;
                map[vp_id].threads[th_id + ht_id]->ht[0] = ht_id;
                if( 0 == --nbcores ) {
                    map[vp_id].nbthreads = th_id + ht_id + 1;
                    nbvp = vp_id + 1;  /* Update the number of valid VP */
                    goto complete_and_return;
                }
            }
            core_id++;
        }
    }
  complete_and_return:
    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_datamap;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_datamap;
    vpmap_get_core_affinity = vpmap_get_core_affinity_datamap;

    return 0;
#else
    (void)nbcores;
    return -1;
#endif
}

int vpmap_init_from_file(const char *filename)
{
    FILE *f;
    char *line = NULL;
    size_t nline = 0;
    int rank = 0;
    int nbth = 1, nbcores, c, v;

    if( nbvp != -1 ) {
        vpmap_nb_total_threads = -1;
        return -1;
    }

#if defined(PARSEC_HAVE_HWLOC)
    nbht = parsec_hwloc_get_ht();
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    f = fopen(filename, "r");
    if( NULL == f ) {
        parsec_warning("File open %s: %s (default thread binding).", filename, strerror(errno));
        return -1;
    }

    nbvp = 0;

    char * th_arg = NULL;
    char * binding = NULL;
    long int tgt_rank = -1;

#if defined(PARSEC_HAVE_MPI)
    int mpi_is_on;
    MPI_Initialized(&mpi_is_on);
    if(mpi_is_on) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
#endif
    /* Count the number of line describing a VP for the process rank */
    while( getline(&line, &nline, f) != -1 ) {
        if( NULL != strchr(line, ':') && (line[0] != ':')) {
            tgt_rank = strtol(line, NULL,0);
            if ( tgt_rank == rank ){
                nbvp++;
            }
        } else if( line[0] == ':' ){
            /* no target proc specified, applies to all. */
            nbvp++;
        } else {
            parsec_warning("malformed line %s in vpmap description %s.", line, filename);
        }
    }


    if( 0 == nbvp ) {
        /* If no description is available for the process, create a single monothread VP */
        parsec_inform("No VP parameter for the process %i: create a single VP (single thread, unbound)", rank);
        nbvp = 1;
        map = (vpmap_t*)malloc(sizeof(vpmap_t));
        map[0].nbthreads = 1;
        map[0].threads = (vpmap_thread_t**)malloc(sizeof(vpmap_thread_t*));

#if defined(PARSEC_HAVE_HWLOC)
        nbcores = parsec_hwloc_nb_real_cores();
#else
        nbcores = DEFAULT_NB_CORE;
#endif
        map[0].threads[0] = (vpmap_thread_t*)malloc((nbcores-1) * sizeof(int) + sizeof(vpmap_thread_t));
        map[0].threads[0]->nbcores = nbcores;
        for(c = 0; c < nbcores; c++) {
            map[0].threads[0]->cores[c] = c;
        }

        vpmap_nb_total_threads = nbcores;
    } else {
        /* We have some VP descriptions */
        map = (vpmap_t*)malloc(nbvp * sizeof(vpmap_t));

        rewind(f);
        v = 0;
        vpmap_nb_total_threads  = 0;
        while( getline(&line, &nline, f) != -1 ) {
            if( NULL != strchr(line, ':') ) {
                if (line[0] == ':')
                    tgt_rank=0;
                else
                    tgt_rank = strtod(line, NULL);

                if ( tgt_rank == rank ){
                    nbth=0;
                    if( NULL != (th_arg = strchr(line, ':'))) {
                        /* skip the colon and treat the thread number argument */
                        th_arg++;
                        nbth = (int) strtod(th_arg, NULL);
                        if( nbth <= 0 )
                            nbth=1;

                        vpmap_nb_total_threads += nbth;

                        map[v].nbthreads = nbth;
                        map[v].threads = (vpmap_thread_t**)calloc(nbth, sizeof(vpmap_thread_t*));

                        /* skip the colon and treat the binding argument */
                        if( NULL != (binding = strchr(th_arg, ':'))) {
                            binding++;
                            parse_binding_parameter(v, nbth, binding);

                        } else {
                            printf("[%i] No binding specified for threads of the VP %i \n", rank, v);
                        }
                    }
                    v++;
                }
            }
        }
    }

    fclose(f);

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_datamap;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_datamap;
    vpmap_get_core_affinity = vpmap_get_core_affinity_datamap;

    return 0;
}

int vpmap_init_from_parameters(int _nbvp, int _nbthreadspervp, int _nbcores)
{
    if( nbvp != -1 ||
        nbthreadspervp != -1 ||
        nbcores != -1 ) {
        vpmap_nb_total_threads = -1;
        return -1;
    }

#if defined(PARSEC_HAVE_HWLOC)
    nbht = parsec_hwloc_get_ht();
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    nbcores = _nbcores;
    nbthreadspervp = _nbthreadspervp;
    nbvp = _nbvp;

    vpmap_nb_total_threads = nbvp * nbthreadspervp;

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
    vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;
    return 0;
}

int vpmap_init_from_flat(int _nbcores)
{
    if( nbvp != -1 ||
        nbthreadspervp != -1 ||
        nbcores != -1 ) {
        vpmap_nb_total_threads = -1;
        return -1;
    }

#if defined(PARSEC_HAVE_HWLOC)
    nbht = parsec_hwloc_get_ht();
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    nbvp = 1;
    nbcores = _nbcores/nbht;
    nbthreadspervp = _nbcores;

    vpmap_nb_total_threads = nbvp * nbthreadspervp;

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
    vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;
    return 0;
}

void vpmap_display_map(void) {
    int v, t, c;
    char *cores = NULL, *ht = NULL, *tmp;
    int *dcores, *dht;
    int rc;

    parsec_inform( "Virtual Process Map ...");
    if( -1 == nbvp ) {
        parsec_inform("   Map undefined");
        return;
    }

#if defined(PARSEC_HAVE_HWLOC)
    nbht = parsec_hwloc_get_ht();
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    parsec_inform("Map with %d Virtual Processes", nbvp);
    for(v = 0; v < nbvp; v++) {
        parsec_inform("   Virtual Process of index %d has %d threads",
                     v, vpmap_get_nb_threads_in_vp(v) );
        for(t = 0; t < vpmap_get_nb_threads_in_vp(v); t++) {
            dcores = (int*)malloc(vpmap_get_nb_cores_affinity(v, t) * sizeof(int));
            dht = (int*)malloc(vpmap_get_nb_cores_affinity(v, t) * sizeof(int));
            vpmap_get_core_affinity(v, t, dcores, dht);
            rc = asprintf(&cores, "%d", dcores[0]);
            assert(rc!=-1); (void)rc;

            if(nbht > 1)
                rc = asprintf(&ht, " (ht %d)", dht[0]);
            else
                rc = asprintf(&ht, " ");
            assert(rc != -1);
            for( c = 1; c < vpmap_get_nb_cores_affinity(v, t); c++) {
                tmp=cores;
                rc = asprintf(&cores, "%s, %d", tmp, dcores[c]);
                assert(rc!=-1); (void)rc;
                free(tmp);
            }
            free(dcores);
            free(dht);

            parsec_inform("    Thread %d of VP %d can be bound on cores %s %s",
                         t, v, cores, ht);
            free(cores);
            free(ht);
        }
    }
}


int parse_binding_parameter(int vp, int nbth, char * binding)
{
 #if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    char* option = binding;
    char* position;
    int t, ht;

    assert(NULL != option);

    int nb_real_cores = parsec_hwloc_nb_real_cores();

    /* Parse  hexadecimal mask, range expression of core list expression */
    if( NULL != (position = strchr(option, 'x')) ) {
        /* The parameter is a hexadecimal mask */

        position++; /* skip the x */

        /* convert the mask into a bitmap (define legal core indexes) */
        unsigned long mask = strtoul(position, NULL, 16);
        if (mask < 1)
            parsec_fatal("P %i: empty binding mask", vp);
        hwloc_cpuset_t binding_mask = hwloc_bitmap_alloc();
        hwloc_bitmap_from_ulong(binding_mask, mask);

#if defined(PARSEC_DEBUG_NOISIER)
        {
            char *str = NULL;
            hwloc_bitmap_asprintf(&str,  binding_mask);
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "VP %i : binding of the %i threads defined by the mask %s", vp, nbth, str);
            free(str);
        }
#endif /* defined(PARSEC_DEBUG_NOISIER) */

        int core=-1, prev=-1;
#if defined(PARSEC_HAVE_HWLOC)
        nbht = parsec_hwloc_get_ht();
#endif  /* defined(PARSEC_HAVE_HWLOC) */

        /* extract a single core per thread (round-robin) */
        for( t=0; t<nbth; t+=nbht ) {
            core = hwloc_bitmap_next(binding_mask, prev);
            if( core == -1 || core > nb_real_cores ) {
                prev = -1;
                core = hwloc_bitmap_next(binding_mask, prev);
                parsec_warning("Several thread of the VP number %i will be bound on the same core", vp);
            }
            assert(core != -1);

            for (ht=0; ht < nbht ; ht++){
                map[vp].threads[t+ht] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                map[vp].threads[t+ht]->nbcores = 1;
                map[vp].threads[t+ht]->cores[0] = core;
                if (nbht > 1)
                    map[vp].threads[t+ht]->ht[0] = ht;
                else
                    map[vp].threads[t+ht]->ht[0] = -1;
            }
            prev=core;
        }
        hwloc_bitmap_free(binding_mask);
    } else if( NULL != (position = strchr(option, ';'))) {
        /* The parameter is a range expression such as [start];[end];[step] */

        int arg;
        int start = 0, step = 1;
        int end=nb_real_cores-1;
        if( position != option ) {
            /* we have a starting position */
            arg = strtol(option, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                start = strtol(option, NULL, 10);
            else
                parsec_warning("binding start core not valid (restored to default value)");
        }
        position++;  /* skip the ; */
        if( '\0' != position[0] ) {
            /* check for the ending position */
            if( ';' != position[0] ) {
                arg = strtol(position, &position, 10);
                if( (arg < nb_real_cores) && (arg > -1) )
                    end = arg;
                else
                    parsec_warning("binding end core not valid (restored to default value)");
            }
            position = strchr(position, ';');  /* find the step */
        }
        if( NULL != position )
            position++;  /* skip the ; directly into the step */
        if( (NULL != position) && ('\0' != position[0]) ) {
            arg = strtol(position, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                step = arg;
            else
                parsec_warning("binding step not valid (restored to default value)");
        }

        if( start > end ) {
            parsec_warning("Invalid range: start > end (end restored to default value)");
            end=nb_real_cores-1;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "binding defined by core range [%d;%d;%d]\n", start, end, step);

        /* define the core according to the trio start/end/step */
        {
            int where = start, skip = 1;

            for( t = 0; t < nbth; t+=nbht ) {
                for (ht=0; ht < nbht ; ht++){
                    map[vp].threads[t+ht] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                    map[vp].threads[t+ht]->nbcores = 1;
                    map[vp].threads[t+ht]->cores[0] = where;
                    if (nbht > 1)
                        map[vp].threads[t+ht]->ht[0] = ht;
                    else
                        map[vp].threads[t+ht]->ht[0] = -1;
                }

                where += step;
                if( where > end ) {
                    where = start + skip;
                    skip++;
                    if ( where > end )
                        break;

                    if((skip > step) && (t < (nb_real_cores - 1))) {
                        parsec_warning("No more available core to bind according to the range. The remaining %d threads are not bound", nbth-(t*nbht));
                        int th;
                        for( th = t+nbht; th < nbth;  th++) {
                            map[vp].threads[th] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                            map[vp].threads[th]->nbcores = 1;
                            map[vp].threads[th]->cores[0] = -1;
                            map[vp].threads[th]->ht[0] = -1;

                        }
                        break;
                    }
                }
            }
        }
    } else {
        /* List of cores (binding in order) */
        int core_tab[nbth];
        memset(core_tab, -1, sizeof(int)*nbth);
        int cmp=0;
        int arg, next_arg;

        /* Parse the list. Store the in order cores in core_tab up to nbth.
           If the list is too short, the remaining threads won't be bound  */
        while( option != NULL && option[0] != '\0'  && cmp < nbth ) {

            /* first core of the remaining list to parse*/
            arg = (int) strtol(option, &option, 10);
            if( (arg < nb_real_cores) && (arg > -1) ) {
                core_tab[cmp]=arg;
                cmp++;

            } else {
                parsec_warning("binding core #%i not valid (must be between 0 and %i (nb_core-1)", arg, nb_real_cores-1);
            }

            if( NULL != (position = strpbrk(option, ",-"))) {
                /* parse a core range */
                if( position[0] == '-' ) {
                    /* core range */
                    position++;
                    next_arg = (int) strtol(position, &position, 10);
                    for(t=arg+1; t<=next_arg; t++)
                        if( (t < nb_real_cores) && (t > -1) ) {
                            core_tab[cmp]=t;
                            cmp++;
                            if (cmp == nbth)
                                break;
                        } else {
                            parsec_warning("binding core #%i not valid (must be between 0 and %i (nb_core-1)", t, nb_real_cores-1);
                        }
                }
            }

            /* next potential argument is following a comma */
            option = strchr(option, ',');
            if( NULL != option)
                option++;   /* skip the comma */
        }

#if defined(PARSEC_DEBUG_NOISIER)
        char tmp[128];
        size_t offset = 0;

        for( t = 0; t < nbth; t++ ) {
            if( -1 == core_tab[t] )
                break;
            offset += sprintf(tmp + offset, "%i ", core_tab[t]);
            if( offset > (sizeof(tmp)-4)){
                sprintf(tmp+offset, "...");
                break;
            }
         }
        PARSEC_DEBUG_VERBOSE( 20, parsec_debug_output, "binding defined by the parsed list: %s ", tmp);
#endif /* PARSEC_DEBUG_NOISIER */

        int c=0;
        for( t = 0; t < nbth; t+=nbht ) {
            for (ht=0; ht < nbht ; ht++){
                map[vp].threads[t+ht] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                map[vp].threads[t+ht]->nbcores = 1;
                map[vp].threads[t+ht]->cores[0] = core_tab[c];
                if (nbht > 1 && core_tab[c] > -1 )
                    map[vp].threads[t+ht]->ht[0] = ht;
                else
                    map[vp].threads[t+ht]->ht[0] = -1;
            }
            c++;
        }
    }
    return 0;
#else
    (void)vp; (void)nbth; (void)binding;
    parsec_warning("the binding defined has been ignored (requires a build with HWLOC with bitmap support).");
    return -1;
#endif /* PARSEC_HAVE_HWLOC && PARSEC_HAVE_HWLOC_BITMAP */
}
