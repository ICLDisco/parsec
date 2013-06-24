#include "dague_config.h"

#include <string.h>
#include <assert.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#include "dague_hwloc.h"
#include "vpmap.h"
#include "debug.h"

#define DEFAULT_NB_CORE 128
#define MAX_STR_SIZE 12
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


static vpmap_t *map = NULL;     /**< used by the from_file and from_affinity only */
static int nbvp = -1;
static int nbht = 1;
static int nbthreadspervp = -1; /**< used by the from_parameters only */
static int nbcores = -1;        /**< used by the from_parameters only */

static int vpmap_get_nb_threads_in_vp_parameters(int vp);
static int vpmap_get_nb_cores_affinity_parameters(int vp, int thread);
static void vpmap_get_core_affinity_parameters(int vp, int thread, int *cores, int *ht);

static int parse_binding_parameter(int vp, int nbth, char * binding);

/* The parameters variant is used while no init has been called,
 * as they return -1 in this case for any call
 */
vpmap_get_nb_threads_in_vp_t vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
vpmap_get_nb_cores_affinity_t vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
vpmap_get_core_affinity_t vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;

/* int parse_binding_parameter(void * optarg, dague_context_t* context, */
/*                             __dague_temporary_thread_initialization_t* startup); */


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
    int nb_real_cores = DEFAULT_NB_CORE;
#if defined(HAVE_HWLOC)
    dague_hwloc_init();
    nb_real_cores = dague_hwloc_nb_real_cores();
    nbht = dague_hwloc_get_ht();
#endif /* HAVE_HWLOC */
    *cores = ((vp * nbthreadspervp + thread) / nbht) % nbcores % nb_real_cores;
    if (nbht > 2 )
        *ht = (vp * nbthreadspervp + thread) % nbht;
    else
        *ht = -1;
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

int vpmap_init_from_hardware_affinity(void)
{
#if defined(HAVE_HWLOC)
    int v, t, c, ht;

    dague_hwloc_init();

    /* Compute the number of VP according to the number of objects at the
     * lowest level between sockets and NUMA nodes */
    int level = dague_hwloc_core_first_hrwd_ancestor_depth();
    nbvp = dague_hwloc_get_nb_objects(level);
    nbht = dague_hwloc_get_ht();

    if (nbvp > 0 ) {
        map = (vpmap_t*)malloc(nbvp * sizeof(vpmap_t));

        /* Define the VP map:
         * threads are distributed in order on the cores (hwloc numbering, ensure locality)
         */
        c=0;
        for(v = 0; v < nbvp; v++) {
            nbthreadspervp = dague_hwloc_nb_cores_per_obj(level, v)*nbht;
            map[v].nbthreads = nbthreadspervp;
            map[v].threads = (vpmap_thread_t**)calloc(nbthreadspervp, sizeof(vpmap_thread_t*));

            for(t = 0; t < nbthreadspervp; t+=nbht) {
                for (ht=0; ht < nbht ; ht++){
                    map[v].threads[t+ht] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                    map[v].threads[t+ht]->nbcores = 1;
                    map[v].threads[t+ht]->cores[0] = c;
                    if (nbht > 1)
                        map[v].threads[t+ht]->ht[0] = ht;
                    else
                        map[v].threads[t+ht]->ht[0] = -1;
                }
                c++;
            }
        }

        vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_datamap;
        vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_datamap;
        vpmap_get_core_affinity = vpmap_get_core_affinity_datamap;

        return 0;
    }else{
        vpmap_init_from_flat(dague_hwloc_nb_real_cores());
        return 0;
    }
#else
    return -1;
#endif

}

int vpmap_init_from_file(const char *filename)
{
    FILE *f;
    char *line = NULL;
    size_t nline = 0;
    int rank = 0;
    int nbth, nbcores, c, v;

    if( nbvp != -1 )
        return -1;

    nbht = dague_hwloc_get_ht();

    f = fopen(filename, "r");
    if( NULL == f ) {
        STATUS(("File %s can't be open (default thread binding).\n", filename));
        return -1;
    }

    nbvp = 0;

    char * th_arg = NULL;
    char * binding = NULL;
#if defined(HAVE_MPI)
    double mpi_num = -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* Count the number of line describing a VP for the process rank */
    while( getline(&line, &nline, f) != -1 ) {
        if( NULL != strchr(line, ':') && (line[0] != ':')) {
            mpi_num = strtod(line, NULL);
              if ( mpi_num == rank ){
                nbvp++;
            }
        }else if( (line[0] == ':') && (rank == 0) ){
            nbvp++;
        }
    }
#else
    /* Each line descripe a VP */
    while( getline(&line, &nline, f) != -1 ) {
        if( NULL != strchr(line, ':')) {
            nbvp++;
        }
    }
#endif


    if( nbvp == 0 ) {
        /* If no description is available for the MPI process, create a single monothread VP */
        STATUS(("No VP parameter for the MPI process %i: create a single VP (monothread, unbound)\n", rank));
        nbvp=1;
        map = (vpmap_t*)malloc(sizeof(vpmap_t));
        map[0].nbthreads = 1;
        map[0].threads = (vpmap_thread_t**)malloc(sizeof(vpmap_thread_t*));

#if defined(HAVE_HWLOC)
        dague_hwloc_init();
        nbcores = dague_hwloc_nb_real_cores();
#else
        nbcores = DEFAULT_NB_CORE;
#endif
        map[0].threads[0] = (vpmap_thread_t*)malloc((nbcores-1) * sizeof(int) + sizeof(vpmap_thread_t));
        map[0].threads[0]->nbcores = nbcores;
        for(c = 0; c < nbcores; c++) {
            map[0].threads[0]->cores[c] = c;
        }
    } else {
        /* We have some VP descriptions */
        map = (vpmap_t*)malloc(nbvp * sizeof(vpmap_t));

        rewind(f);
        v = 0;
        while( getline(&line, &nline, f) != -1 ) {
            if( NULL != strchr(line, ':') ) {
#if defined(HAVE_MPI)
                if (line[0] == ':')
                    mpi_num=0;
                else
                    mpi_num = strtod(line, NULL);

                if ( mpi_num == rank ){
#endif
                    nbth=0;
                    if( NULL != (th_arg = strchr(line, ':'))) {
                        /* skip the colon and treat the thread number argument */
                        th_arg++;
                        nbth = (int) strtod(th_arg, NULL);
                        if( nbth <= 0 )
                            nbth=1;

                        map[v].nbthreads = nbth;
                        map[v].threads = (vpmap_thread_t**)calloc(nbth, sizeof(vpmap_thread_t*));

                        /* skip the colon and treat the binding argument */
                        if( NULL != (binding = strchr(th_arg, ':'))) {
                            binding++;
                            parse_binding_parameter(v, nbth, binding);

                        } else {
#if defined(HAVE_MPI)
                            printf("[%i] No binding specified for threads of the VP %i \n", rank, v);
#else
                            printf("No binding specified for threads of the VP %i \n", v);
#endif
                        }
                    }
                    v++;
#if defined(HAVE_MPI)
                }
#endif
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
        nbcores != -1 )
        return -1;

    nbht = dague_hwloc_get_ht();

    nbcores = _nbcores;
    nbthreadspervp = _nbthreadspervp;
    nbvp = _nbvp;

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
    vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;
    return 0;
}

int vpmap_init_from_flat(int _nbcores)
{
    if( nbvp != -1 ||
        nbthreadspervp != -1 ||
        nbcores != -1 )
        return -1;

    nbht = dague_hwloc_get_ht();

    nbvp = 1;
    nbcores = _nbcores/nbht;
    nbthreadspervp = _nbcores;

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_parameters;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_parameters;
    vpmap_get_core_affinity = vpmap_get_core_affinity_parameters;
    return 0;
}

void vpmap_display_map(FILE *out)
{
    int rank = 0;
    int v, t, c;
    char *cores = NULL, *ht = NULL, *tmp;
    int *dcores, *dht;
#if defined(HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    fprintf(out, "# [%d] Virtual Process Map ...\n", rank);
    if( -1 == nbvp ) {
        fprintf(out, "# [%d]   Map undefined\n", rank);
        return;
    }

    nbht = dague_hwloc_get_ht();

    fprintf(out, "# [%d]  Map with %d Virtual Processes\n", rank, nbvp);
    for(v = 0; v < nbvp; v++) {
        fprintf(out, "# [%d]  Virtual Process of index %d has %d threads\n",
                rank, v, vpmap_get_nb_threads_in_vp(v) );
        for(t = 0; t < vpmap_get_nb_threads_in_vp(v); t++) {
            dcores = (int*)malloc(vpmap_get_nb_cores_affinity(v, t) * sizeof(int));
            dht = (int*)malloc(vpmap_get_nb_cores_affinity(v, t) * sizeof(int));
            vpmap_get_core_affinity(v, t, dcores, dht);
            asprintf(&cores, "%d", dcores[0]);
            if(nbht > 1)
                asprintf(&ht, " (ht %d)", dht[0]);
            else
                asprintf(&ht, "");
            for( c = 1; c < vpmap_get_nb_cores_affinity(v, t); c++) {
                tmp=cores;
                asprintf(&cores, "%s, %d", tmp, dcores[c]);
                free(tmp);
            }
            free(dcores);

            fprintf(out, "# [%d]    Thread %d of VP %d can be bound on cores %s %s\n",
                    rank, t, v, cores, ht);
            free(cores);
        }
    }
}


int parse_binding_parameter(int vp, int nbth, char * binding)
{
 #if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    char* option = binding;
    char* position;
    int t;

    dague_hwloc_init();
    int nb_real_cores = dague_hwloc_nb_real_cores();

    /* Parse  hexadecimal mask, range expression of core list expression */
    if( NULL != (position = strchr(option, 'x')) ) {
        /* The parameter is a hexadecimal mask */
        position++; /* skip the x */

        /* convert the mask into a bitmap (define legal core indexes) */
        unsigned long mask = strtoul(position, NULL, 16);
        if (mask < 1)
            ERROR(("P %i: empty binding mask\n", vp));
        hwloc_cpuset_t binding_mask = hwloc_bitmap_alloc();
        hwloc_bitmap_from_ulong(binding_mask, mask);

#if defined(DAGUE_DEBUG_VERBOSE2)
        {
            char *str = NULL;
            hwloc_bitmap_asprintf(&str,  binding_mask);
            DEBUG2(("VP %i : binding of the %i threads defined by the mask %s\n", vp, nbth, str));
            printf("VP %i : binding of the %i threads defined by the mask %s\n", vp, nbth, str);
            free(str);
        }
#endif /* DAGUE_DEBUG_VERBOSE2 */

        int core=-1, prev=-1;
        /* extract a single core per thread (round-robin) */
        for( t=0; t<nbth; t++ ) {
            core = hwloc_bitmap_next(binding_mask, prev);
            if( core == -1 || core > nb_real_cores ) {
                prev = -1;
                core = hwloc_bitmap_next(binding_mask, prev);
                WARNING(("Several thread of the VP number %i will be bound on the same core\n", vp));
            }
            assert(core != -1);

            map[vp].threads[t] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
            map[vp].threads[t]->nbcores = 1;
            map[vp].threads[t]->cores[0] = core;
            prev++;
        }

        hwloc_bitmap_free(binding_mask);
    } else if( NULL != (position = strchr(option, ':'))) {
        /* The parameter is a range expression such as [start]:[end]:[step] */
        int arg;
        int start = 0, step = 1;
        int end=nb_real_cores-1;
        if( position != option ) {
            /* we have a starting position */
            arg = strtol(option, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                start = strtol(option, NULL, 10);
            else
                WARNING(("binding start core not valid (restored to default value)"));
        }
        position++;  /* skip the : */
        if( '\0' != position[0] ) {
            /* check for the ending position */
            if( ':' != position[0] ) {
                arg = strtol(position, &position, 10);
                if( (arg < nb_real_cores) && (arg > -1) )
                    end = arg;
                else
                    WARNING(("binding end core not valid (restored to default value)\n"));
            }
            position = strchr(position, ':');  /* find the step */
        }
        if( NULL != position )
            position++;  /* skip the : directly into the step */
        if( (NULL != position) && ('\0' != position[0]) ) {
            arg = strtol(position, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                step = arg;
            else
                WARNING(("binding step not valid (restored to default value)\n"));
        }
        DEBUG3(("binding defined by core range [%d:%d:%d]\n", start, end, step));
        printf("binding defined by core range [%d:%d:%d]\n", start, end, step);

        /* define the core according to the trio start/end/step */
        {
            int where = start, skip = 1;
            for( t = 0; t < nbth; t++ ) {
                map[vp].threads[t] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                map[vp].threads[t]->nbcores = 1;
                map[vp].threads[t]->cores[0] = where;

                where += step;
                if( where > end ) {
                    where = start + skip;
                    skip++;
                    if((skip > step) && (t < (nb_real_cores - 1))) {
                        STATUS(( "No more available core to bind according to the range. The remaining %d threads are not bound\n", nbth -1-t));
                        int th;
                        for( th = t+1; th < nbth; th++ ) {
                            map[vp].threads[th] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                            map[vp].threads[th]->nbcores = 1;
                            map[vp].threads[th]->cores[0] = -1;
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
        if( NULL != option ) {
            while( option != NULL && option[0] != '\0'  && cmp < nbth ) {

                /* first core of the remaining list to parse*/
                arg = (int) strtol(option, &option, 10);
                if( (arg < nb_real_cores) && (arg > -1) ) {
                    core_tab[cmp]=arg;
                    cmp++;

                } else {
                    WARNING(("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n", arg, nb_real_cores-1));
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
                                WARNING(("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n", t, nb_real_cores-1));
                            }
                    }
                }

                /* next potential argument is following a comma */
                option = strchr(option, ',');
                if( NULL != option)
                    option++;   /* skip the comma */
            }
        }

#if defined(DAGUE_DEBUG_VERBOSE)
        char tmp[MAX_STR_SIZE];
        char* str = tmp;
        size_t offset;
        int length=0;

        for(t=0; t<nbth; t++){
            if( core_tab[t]==-1 )
                break;
            offset = sprintf(str, "%i ", core_tab[t]);
            length += offset;
            if( length > MAX_STR_SIZE-3){
                sprintf(str, "...");
                break;
            }
            str += offset;
         }
        DEBUG(( "binding defined by the parsed list: %s \n", tmp));
#endif /* DAGUE_DEBUG_VERBOSE */

        for(t=0; t<nbth; t++){
            map[vp].threads[t] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
            map[vp].threads[t]->nbcores = 1;
            map[vp].threads[t]->cores[0] = core_tab[t];
        }
    }
    return 0;
#else
    (void)vp; (void)nbth; (void)binding;
	WARNING(("the binding defined has been ignored (requires a build with HWLOC with bitmap support).\n"));
    return -1;
#endif /* HAVE_HWLOC && HAVE_HWLOC_BITMAP */
}
