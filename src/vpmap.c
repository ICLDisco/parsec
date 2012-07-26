#include "dague_config.h"

#include <string.h>
#include <assert.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#include "dague_hwloc.h"
#include "vpmap.h"

#define DEFAULT_NB_CORE 128

/**
 * These structures are used by the from_hardware and from_file
 * to store the whole vp map
 */
typedef struct {
    int nbcores;
    int cores[1];
} vpmap_thread_t;

typedef struct {
    int nbthreads;
    vpmap_thread_t **threads;
} vpmap_t;

static vpmap_t *map = NULL;     /**< used by the from_file and from_affinity only */
static int nbvp = -1;
static int nbthreadspervp = -1; /**< used by the from_parameters only */
static int nbcores = -1;        /**< used by the from_parameters only */

static int vpmap_get_nb_threads_in_vp_parameters(int vp);
static int vpmap_get_nb_cores_affinity_parameters(int vp, int thread);
static void vpmap_get_core_affinity_parameters(int vp, int thread, int *cores);

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

static void vpmap_get_core_affinity_parameters(int vp, int thread, int *cores)
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
#endif /* HAVE_HWLOC */
    *cores = (vp * nbthreadspervp + thread) % nbcores % nb_real_cores;
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

static void vpmap_get_core_affinity_datamap(int vp, int thread, int *cores)
{
    if( (vp < 0) ||
        (vp >= nbvp) ||
        (map == NULL) ||
        (thread < 0) ||
        (thread >= map[vp].nbthreads ) )
        return;
    memcpy(cores, map[vp].threads[thread]->cores, map[vp].threads[thread]->nbcores * sizeof(int));
}

int vpmap_init_from_hardware_affinity(void)
{
#if defined(HAVE_HWLOC)
    int nblevels;
    int m, v, p, t;
    int *mperc;

    dague_hwloc_init();

    if( (nblevels = dague_hwloc_nb_levels()) <= 0 ||
        (nbcores = dague_hwloc_nb_real_cores()) <= 0 )
        return -1;

    /* Take the maximal level */
    if( nblevels > 1 )
        nblevels = 1;

    mperc = (int*)calloc(sizeof(int), nbcores);
    nbvp = 0;

    for(p = 0; p < nbcores; p++) {
        m = dague_hwloc_master_id(nblevels, p);
        assert( m >= 0 && m < nbcores );
        if( mperc[m] == 0 ) {
            nbvp++;
        }
        mperc[m]++;
    }

    map = (vpmap_t*)malloc(nbvp * sizeof(vpmap_t));

    v = 0;
    for(p = 0; p < nbcores; p++) {
        if( mperc[p] != 0 ) {
            map[v].nbthreads = mperc[p];
            map[v].threads = (vpmap_thread_t**)calloc(mperc[p], sizeof(vpmap_thread_t*));
            t = 0;
            for(m = 0; m < nbcores; m++) {
                if( dague_hwloc_master_id(nblevels, m) == p ) {
                    map[v].threads[t] = (vpmap_thread_t*)malloc(sizeof(vpmap_thread_t));
                    map[v].threads[t]->nbcores = 1;
                    map[v].threads[t]->cores[0] = m;
                    t++;
                }
            }
            v++;
        }
    }

    free(mperc);

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_datamap;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_datamap;
    vpmap_get_core_affinity = vpmap_get_core_affinity_datamap;

    return 0;
#else
    return -1;
#endif
}

int vpmap_init_from_file(const char *filename)
{

    FILE *f;
    char *line = NULL;
    size_t nline = 0;
    int nbth, nbcores, c, v;

    if( nbvp != -1 )
        return -1;

    f = fopen(filename, "r");
    if( NULL == f ) {
        return -1;
    }
    nbvp = 1;
    while( getline(&line, &nline, f) != -1 ) {
        if( !strcmp(line, "\n") ) {
            nbvp++;
        }
    }

    map = (vpmap_t*)malloc(nbvp * sizeof(vpmap_t));

    rewind(f);
    nbth = 0;
    v = 0;
    while( getline(&line, &nline, f) != -1 ) {
        if( !strcmp(line, "\n") ) {
            map[v].nbthreads = nbth;
            map[v].threads = (vpmap_thread_t**)calloc(nbth, sizeof(vpmap_thread_t*));
            nbth = 0;
            v++;
        } else {
            nbth++;
        }
    }
    map[v].nbthreads = nbth;
    map[v].threads = (vpmap_thread_t**)calloc(nbth, sizeof(vpmap_thread_t*));

    rewind(f);
    v = 0;
    nbth = 0;
    while( getline(&line, &nline, f) != -1 ) {
        if( !strcmp(line, "\n") ) {
            nbth = 0;
            v++;
        } else {
            nbcores = 1;
            for(c = 0; c < (int)strlen(line); c++)
                if( line[c] == ' ' )
                    nbcores++;
            map[v].threads[nbth] = (vpmap_thread_t*)malloc( (nbcores-1) * sizeof(int) + sizeof(vpmap_thread_t) );
            map[v].threads[nbth]->nbcores = nbcores;

            nbcores = 0;
            sscanf(line, "%d", &map[v].threads[nbth]->cores[nbcores]);
            for(c = 0; c < (int)strlen(line); c++) {
                if( line[c] == ' ' ) {
                    nbcores++;
                    sscanf(line + c + 1, "%d", &map[v].threads[nbth]->cores[nbcores]);
                }
            }
            nbth++;
        }
    }

    fclose(f);

    vpmap_get_nb_threads_in_vp = vpmap_get_nb_threads_in_vp_datamap;
    vpmap_get_nb_cores_affinity = vpmap_get_nb_cores_affinity_datamap;
    vpmap_get_core_affinity = vpmap_get_core_affinity_datamap;
    printf(" vpmap_get_core_affinity = \n");
    return 0;
}

int vpmap_init_from_parameters(int _nbvp, int _nbthreadspervp, int _nbcores)
{
    if( nbvp != -1 ||
        nbthreadspervp != -1 ||
        nbcores != -1 )
        return -1;

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

    nbvp = 1;
    nbcores = _nbcores;
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
    char *cores = NULL, *tmp;
     int *dcores;
#if defined(HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    fprintf(out, "# [%d] Virtual Process Map ...\n", rank);
    if( -1 == nbvp ) {
        fprintf(out, "# [%d]   Map undefined\n", rank);
        return;
    }

    fprintf(out, "# [%d]  Map with %d Virtual Processes\n", rank, nbvp);
    for(v = 0; v < nbvp; v++) {
        fprintf(out, "# [%d]  Virtual Process of index %d has %d threads\n",
                rank, v, vpmap_get_nb_threads_in_vp(v) );
        for(t = 0; t < vpmap_get_nb_threads_in_vp(v); t++) {
            dcores = (int*)malloc(vpmap_get_nb_cores_affinity(v, t) * sizeof(int));
            vpmap_get_core_affinity(v, t, dcores);
            asprintf(&cores, "%d", dcores[0]);
            for( c = 1; c < vpmap_get_nb_cores_affinity(v, t); c++) {
                tmp=cores;
                asprintf(&cores, "%s, %d", tmp, dcores[c]);
                free(tmp);
            }
            free(dcores);
            fprintf(out, "# [%d]    Thread %d of VP %d can be bound on cores %s\n",
                    rank, t, v, cores);
            free(cores);
        }
    }
}
