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
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/constants.h"

/* If HWLOC is not available support up to 64 cores */
#define DEFAULT_NB_CORE 64
/**
 * Structures defining the virtual process (VP) maps. If HWLOC is not
 * available we only support a single VP, flat across all the available
 * cores.
 */
typedef struct {
    int nbcores;
    int ht;
    hwloc_cpuset_t cpuset;
} vpmap_thread_t;

typedef struct {
    int nbthreads;
    vpmap_thread_t* threads;
    hwloc_cpuset_t cpuset;
} vpmap_t;

static vpmap_t *parsec_vpmap = NULL;     /**< used by the from_file and from_affinity only */
static int parsec_nbvp = -1;
static int parsec_nbht = 1;
static int parsec_nb_total_threads = 0;
static int parse_binding_parameter(int vp, int nbth, char * binding);

/* PARSEC_VPMAP_INIT_*
 *  Different ways of initializing the vpmap.
 * Forall XXX, YYY,
 *     vpmap_init_XXX cannot be called after a succesful call to vpmap_init_YYY
 */

/**
 * Initialize the vpmap based on the HWLOC hardware locality information. Do not
 * initialize more than the expected number of cores.
 *   Create one thread per core
 *   Create one vp per socket
 *   Bind threads of the same vp on the different cores of the
 *     corresponding socket
 *   Uses hwloc
 * @return PARSEC_SUCCESS if success; -1 if the initialization was not possible.
 */
static int parsec_vpmap_init_from_hardware_affinity(int nbcores);

/**
 * initialize the vpmap using a simple nbvp x nbthreadspervp
 *   approach; and a round-robin distribution of threads among cores.
 */
static int parsec_vpmap_init_from_parameters(int nbvp, int nbthreadspervp, int nbcores);

/**
 * initialize the vpmap using a very simple flat x nbcores approach
 */
static int parsec_vpmap_init_from_flat(int nbcores);

/**
 * initialize the vpmap using a virtual process rank file
 *  Format of the rankfile:
 *  list of integers: cores of thread 0 of vp 0
 *  list of integers: cores of thread 1 of vp 0
 *  ...
 *  blank line: change the vp number
 *  list of integers: cores of thread 0 of vp 1
 *  ...
 */
static int parsec_vpmap_init_from_file(const char *filename);

int parsec_vpmap_get_nb_vp(void)
{
    return parsec_nbvp;
}

int parsec_vpmap_init(char* optarg, int nb_cores )
{
    if( NULL == optarg )  /* build a flat topology */
        goto build_flat_topology;
    /* We accept a vpmap that starts with "display:" as a mean to show the mapping */
    if( !strncmp(optarg, "display", 7 )) {
        parsec_report_binding_issues = 1;
        if( ':' != optarg[strlen("display")] ) {
            parsec_warning("Display thread mapping requested but vpmap argument incorrect "
                           "(must start with display: to print the mapping)");
        } else {
            optarg += strlen("display:");
        }
    }
    if( !strncmp(optarg, "flat", 4) ) {
        /* default case (handled in parsec_init) */
    } else if( !strncmp(optarg, "hwloc", 5) ) {
        parsec_vpmap_init_from_hardware_affinity(nb_cores);
    } else if( !strncmp(optarg, "file:", 5) ) {
        parsec_vpmap_init_from_file(optarg + 5);
    } else if( !strncmp(optarg, "rr:", 3) ) {
        int n, p, co;
        if( sscanf(optarg, "rr:%d:%d:%d", &n, &p, &co) == 3 ) {
            parsec_vpmap_init_from_parameters(n, p, co);
        } else {
            parsec_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
        }
    } else {
        if( '\0' != optarg[0] ) {
            parsec_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
        }
    }
    if( -1 == parsec_vpmap_get_nb_vp() ) {
build_flat_topology:
        parsec_vpmap_init_from_flat(nb_cores);
    }
    /* Consolidate the VP cpuset (hwloc_bitmap_intersects) */
    for( int i = 0; i < parsec_nbvp; i++ ) {
        parsec_vpmap[i].cpuset = HWLOC_ALLOC();
        for( int j = 0; j < parsec_vpmap[i].nbthreads; j++ ) {
            if( parsec_runtime_singlify_bindings > 0 )  /* late singlify */
                hwloc_bitmap_singlify(parsec_vpmap[i].threads[j].cpuset);
            if( HWLOC_INTERSECTS(parsec_vpmap[i].cpuset, parsec_vpmap[i].threads[j].cpuset) ) {
                /* overlap detected, show it to the user */
                if(parsec_report_binding_issues) {
                    parsec_warning("VP[%d] cpuset intersects with thread %d\n, a compute resource is over-committed\n",
                                   i, j);
                    parsec_report_bindings = 1;  /* report binding issues to keep the user informed */
                }
            }
            HWLOC_OR(parsec_vpmap[i].cpuset, parsec_vpmap[i].cpuset, parsec_vpmap[i].threads[j].cpuset);
        }
    }
    return 0;
}

void parsec_vpmap_fini(void)
{
    int v, t;
    if( NULL != parsec_vpmap ) {
        for(v = 0; v < parsec_nbvp; v++) {
            for(t = 0; t < parsec_vpmap[v].nbthreads; t++) {
                HWLOC_FREE(parsec_vpmap[v].threads[t].cpuset);
            }
            free(parsec_vpmap[v].threads);
            HWLOC_FREE(parsec_vpmap[v].cpuset);
        }
        free(parsec_vpmap);
        parsec_vpmap = NULL;
    }
    parsec_nbvp = -1;
}

int parsec_vpmap_get_vp_threads(int vp)
{
    if( (vp < 0) ||
        (vp >= parsec_nbvp) ||
        (NULL == parsec_vpmap) )
        return PARSEC_ERR_BAD_PARAM;
    return parsec_vpmap[vp].nbthreads;
}

int parsec_vpmap_get_vp_thread_cores(int vp, int thread)
{
    if( (vp < 0) ||
        (vp >= parsec_nbvp) ||
        (parsec_vpmap == NULL) ||
        (thread < 0) ||
        (thread >= parsec_vpmap[vp].nbthreads ) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    return parsec_vpmap[vp].threads[thread].nbcores;
}

hwloc_cpuset_t parsec_vpmap_get_vp_thread_affinity(int vp, int thread, int *ht)
{
    if( (vp < 0) ||
        (vp >= parsec_nbvp) ||
        (parsec_vpmap == NULL) ||
        (thread < 0) ||
        (thread >= parsec_vpmap[vp].nbthreads ) )
        return NULL;
    *ht = parsec_vpmap[vp].threads[thread].ht;
    return parsec_vpmap[vp].threads[thread].cpuset;
}

int parsec_vpmap_init_from_hardware_affinity(int nbthreads)
{
#if defined(PARSEC_HAVE_HWLOC)
    int vp_id, th_id, core_id, ht_id, nbthreadspervp;

    /* Compute the number of VP according to the number of objects at the
     * lowest level between sockets and NUMA nodes */
    int level = parsec_hwloc_core_first_hrwd_ancestor_depth();
    parsec_nbvp = parsec_hwloc_get_nb_objects(level);
    parsec_nbht = parsec_hwloc_get_ht();

    if ( parsec_nbvp <= 0 ) {
        return parsec_vpmap_init_from_flat(nbthreads);
    }

    parsec_vpmap = (vpmap_t*)calloc(parsec_nbvp, sizeof(vpmap_t));
    /* Define the VP map:
     * threads are distributed in order on the cores (hwloc numbering, ensure locality)
     */
    core_id = 0;
    parsec_nb_total_threads = 0;

    for( vp_id = 0; vp_id < parsec_nbvp; vp_id++ ) {
        nbthreadspervp = parsec_hwloc_nb_cores_per_obj(level, vp_id) * parsec_nbht;
        if( PARSEC_SUCCESS > nbthreadspervp )
            parsec_fatal("could not determine the number of threads for the VP map");
        parsec_nb_total_threads += nbthreadspervp;

        parsec_vpmap[vp_id].cpuset = parsec_hwloc_cpuset_per_obj(level, vp_id);
        parsec_vpmap[vp_id].nbthreads = nbthreadspervp;
        parsec_vpmap[vp_id].threads   = (vpmap_thread_t*)calloc(nbthreadspervp, sizeof(vpmap_thread_t));

        for( th_id = 0; th_id < nbthreadspervp; th_id += parsec_nbht ) {
            for( ht_id = 0; ht_id < parsec_nbht ; ht_id++ ) {
                parsec_vpmap[vp_id].threads[th_id + ht_id].nbcores = 1;
                parsec_vpmap[vp_id].threads[th_id + ht_id].cpuset = parsec_hwloc_cpuset_per_obj(level+1, core_id);
                parsec_vpmap[vp_id].threads[th_id + ht_id].ht = ht_id;
                if( 0 == --nbthreads ) {
                    parsec_vpmap[vp_id].nbthreads = th_id + ht_id + 1;
                    parsec_nbvp = vp_id + 1;  /* Update the number of valid VP */
                    goto complete_and_return;
                }
            }
            core_id++;
        }
    }
  complete_and_return:
    return PARSEC_SUCCESS;
#else
    (void)nbthreads;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif
}

int parsec_vpmap_init_from_file(const char *filename)
{
    FILE *f;
    char *line = NULL;
    size_t nline = 0;
    int rank = 0, nbth = 1;

    if( parsec_nbvp != -1 ) {  /* cannot overload an existing vpmap */
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    f = fopen(filename, "r");
    if( NULL == f ) {
        parsec_warning("File open %s: %s (default thread binding).", filename, strerror(errno));
        return PARSEC_ERR_NOT_FOUND;
    }

    parsec_nbvp = 0;

    char * th_arg = NULL;
    char * binding = NULL;
    char *local_vpmap = NULL, *next_string, *rest_of_line;
    long int tgt_rank = -1;

#if defined(PARSEC_HAVE_MPI)
    int mpi_is_on;
    MPI_Initialized(&mpi_is_on);
    if(mpi_is_on) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
#endif
    /* Count the number of lines describing a VP for the process rank */
    while( getline(&line, &nline, f) != -1 ) {
        if( NULL != strchr(line, ':') && (line[0] != ':')) {
            tgt_rank = strtol(line, &rest_of_line, 0);
            if ( tgt_rank != rank ) {
                continue;
            }
        } else if( line[0] == ':' ) {
            /* no target proc specified, applies to all. */
        } else {
            parsec_warning("malformed line %s in vpmap description %s.", line, filename);
            continue;
        }
        /* Add the current vpmap description to the local_vpmap */
        parsec_nbvp++;
        if( NULL == local_vpmap ) {
            asprintf(&next_string, "%s\n%s", local_vpmap, rest_of_line);
            free(local_vpmap);
        } else {
            next_string = strdup(rest_of_line);
        }
        local_vpmap = next_string;
    }
    fclose(f);

    if( 0 == parsec_nbvp ) {
        /* If no description is available for the process, create a single monothread VP */
        parsec_inform("No VP parameter for the process %i: create a single VP (single thread, unbound)", rank);
        return parsec_vpmap_init_from_flat(-1);
    }
    /* We have some VP descriptions */
    parsec_vpmap = (vpmap_t*)malloc(parsec_nbvp * sizeof(vpmap_t));

    line = local_vpmap;
    parsec_nb_total_threads  = 0;
    int v = 0;
    while( NULL != line ) {
        next_string = strchr(line, '\n');
        if( NULL != next_string ) {
            *next_string = '\0';  /* add an end of string */
            next_string++;        /* and move to the next element */
        }
        nbth = 0;
        if( NULL != (th_arg = strchr(line, ':'))) {
            /* skip the colon and treat the thread number argument */
            th_arg++;
            nbth = (int) strtod(th_arg, NULL);
            if( nbth <= 0 )
                nbth=1;
            
            parsec_nb_total_threads += nbth;
            
            parsec_vpmap[v].nbthreads = nbth;
            parsec_vpmap[v].threads = (vpmap_thread_t*)calloc(nbth, sizeof(vpmap_thread_t));
            
            /* skip the colon and treat the binding argument */
            if( NULL != (binding = strchr(th_arg, ':'))) {
                binding++;
                parse_binding_parameter(v, nbth, binding);
            } else {
                printf("[%i] No binding specified for threads of the VP %i \n", rank, v);
            }
        }
        v++;
        line = next_string;
    }
    free(local_vpmap);
    return PARSEC_SUCCESS;
}

int parsec_vpmap_init_from_parameters(int _nbvp, int _nbthreadspervp, int _nbcores)
{
    if( parsec_nbvp != -1 ) {  /* do not overload an existing topology */
        return PARSEC_ERR_BAD_PARAM;
    }

    parsec_nbvp = _nbvp;
    assert(0);  /* TODO: finish this */
    parsec_nb_total_threads = _nbvp * _nbthreadspervp;
    (void)_nbcores;
    return PARSEC_SUCCESS;
}

int parsec_vpmap_init_from_flat(int nbthreads)
{
    if( parsec_nbvp != -1 ) {  /* do not overload an existing vpmap */
        return PARSEC_ERR_BAD_PARAM;
    }

    int nbcores = DEFAULT_NB_CORE;
#if defined(PARSEC_HAVE_HWLOC)
    nbcores = parsec_hwloc_nb_real_cores();
#endif
    if( -1 == nbthreads ) nbthreads = nbcores;
    parsec_nbvp = 1;
    parsec_vpmap = (vpmap_t*)malloc(sizeof(vpmap_t));
    parsec_vpmap[0].nbthreads = nbthreads;
    parsec_vpmap[0].threads = (vpmap_thread_t*)calloc(parsec_vpmap[0].nbthreads, sizeof(vpmap_thread_t));

    int step = nbcores / nbthreads;
    if( -1 == parsec_runtime_singlify_bindings )  /* early singlify */
        step = 1;
    for( int id = 0; id < parsec_vpmap[0].nbthreads; id++ ) {
        parsec_vpmap[0].threads[id].nbcores = step;
        parsec_vpmap[0].threads[id].cpuset = HWLOC_ALLOC();
        parsec_vpmap[0].threads[id].ht = 0;
        hwloc_bitmap_set_range(parsec_vpmap[0].threads[id].cpuset, id * step, (id+1) * step - 1);
    }
    parsec_nb_total_threads = nbthreads;
    return PARSEC_SUCCESS;
}

void parsec_vpmap_display_map(void)
{
    char *str = NULL, *pstr = NULL;

    parsec_inform( "Virtual Process Map with %d VPs...", parsec_nbvp);
    if( -1 == parsec_nbvp ) {
        parsec_inform("   Map undefined");
        return;
    }

    for(int v = 0; v < parsec_nbvp; v++) {
        str = parsec_hwloc_convert_cpuset(0, parsec_vpmap[v].cpuset);
        pstr = parsec_hwloc_convert_cpuset(1, parsec_vpmap[v].cpuset);
        parsec_inform("   Virtual Process of index %d has %d threads and logical cpuset %s\n"
                      "           physical cpuset %s\n",
                      v, parsec_vpmap[v].nbthreads, str, pstr );
        free(str);free(pstr);
        for(int t = 0; t < parsec_vpmap[v].nbthreads; t++) {
            str = parsec_hwloc_convert_cpuset(0, parsec_vpmap[v].threads[t].cpuset);
            pstr = parsec_hwloc_convert_cpuset(1, parsec_vpmap[v].threads[t].cpuset);
            parsec_inform("    Thread %d of VP %d can be bound on logical cores %s (physical cores %s)\n",
                          t, v, str, pstr);
            free(str); free(pstr);
        }
    }
}


int parse_binding_parameter(int vp, int nbth, char * binding)
{
 #if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    char* option = binding;
    char* position;
    int t, ht, nbht = parsec_hwloc_get_ht();

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

        /* provide a single core per thread (round-robin) */
        for( t = 0; t < nbth; t += nbht ) {
            core = hwloc_bitmap_next(binding_mask, prev);
            if( core == -1 || core > nb_real_cores ) {
                prev = -1;
                core = hwloc_bitmap_next(binding_mask, prev);
                parsec_warning("Several thread of the VP number %i will be bound on the same core", vp);
            }
            assert(core != -1);

            for (ht=0; ht < nbht ; ht++) {
                parsec_vpmap[vp].threads[t+ht].nbcores = 1;
                parsec_vpmap[vp].threads[t+ht].cpuset = HWLOC_ALLOC();
                hwloc_bitmap_set(parsec_vpmap[vp].threads[t+ht].cpuset, core);
                parsec_vpmap[vp].threads[t+ht].ht = (nbht > 1 ? ht : -1);
            }
            prev = core;
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

            for( t = 0; t < nbth; t += nbht ) {
                for (ht=0; ht < nbht ; ht++){
                    parsec_vpmap[vp].threads[t+ht].nbcores = 1;
                    parsec_vpmap[vp].threads[t+ht].cpuset = HWLOC_ALLOC();
                    HWLOC_SET(parsec_vpmap[vp].threads[t+ht].cpuset, where);
                    parsec_vpmap[vp].threads[t+ht].ht = (nbht > 1 ? ht : -1);
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
                            parsec_vpmap[vp].threads[th].nbcores = 1;
                            parsec_vpmap[vp].threads[t+ht].cpuset = HWLOC_ALLOC();
                            /* nothing set */
                            parsec_vpmap[vp].threads[t+ht].ht = (nbht > 1 ? ht : -1);
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
                parsec_vpmap[vp].threads[t+ht].nbcores = 1;
                parsec_vpmap[vp].threads[t+ht].cpuset = HWLOC_ALLOC();
                HWLOC_SET(parsec_vpmap[vp].threads[t+ht].cpuset, core_tab[c]);
                parsec_vpmap[vp].threads[t+ht].ht = (nbht > 1 && core_tab[c] > -1) ? ht : -1;
            }
            c++;
        }
    }
    return PARSEC_SUCCESS;
#else
    (void)vp; (void)nbth; (void)binding;
    parsec_warning("the binding defined has been ignored (requires a build with HWLOC with bitmap support).");
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif /* PARSEC_HAVE_HWLOC && PARSEC_HAVE_HWLOC_BITMAP */
}
