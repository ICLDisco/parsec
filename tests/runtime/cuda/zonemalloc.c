#include "parsec.h"
#include "parsec/utils/zone_malloc.h"

#define NUM_SEGMENTS 128


int main(int argc, char **argv) {
    parsec_context_t *parsec = NULL;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
#endif /* DISTRIBUTED */

    /* Initialize PaRSEC */
    parsec = parsec_init(1, &argc, &argv);
    if( NULL == parsec ) {
        /* Failed to correctly initialize. In a correct scenario report*/
         /* upstream, but in this particular case bail out.*/
        exit(-1);
    }

    char** segments = calloc(NUM_SEGMENTS, sizeof(char*));

    char *base = malloc(NUM_SEGMENTS*512);
    zone_malloc_t *gdata;
    gdata = zone_malloc_init(base, NUM_SEGMENTS, 512);
    assert(gdata != NULL);

    /* allocate and free all segments same size */
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        segments[i] = zone_malloc(gdata, 512);
        assert(segments[i] != NULL);
    }
    zone_debug(gdata, 0, 0, "sequential alloc: ");

    /* free all segments sequentially */
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        zone_free(gdata, segments[i]);
        segments[i] = NULL;
    }
    zone_debug(gdata, 0, 0, "sequential free: ");

    /* allocate all segments same size */
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        segments[i] = zone_malloc(gdata, 512);
        assert(segments[i] != NULL);
    }
    zone_debug(gdata, 0, 0, "stride alloc: ");

    /* free segments with stride */
    for (int i = 0; i < NUM_SEGMENTS; i += 2) {
        zone_free(gdata, segments[i]);
        segments[i] = NULL;
    }
    zone_debug(gdata, 0, 0, "stried free1: ");
    for (int i = 1; i < NUM_SEGMENTS; i += 2) {
        zone_free(gdata, segments[i]);
        segments[i] = NULL;
    }
    zone_debug(gdata, 0, 0, "stried free2: ");

    /* allocate segments with 2 different sizes (256, 512) */
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        segments[i] = zone_malloc(gdata, 512/((i%2)+1));
        assert(segments[i] != NULL);
    }

    /* free them in reverse order */
    for (int i = NUM_SEGMENTS-1; i > 0; i--) {
        zone_free(gdata, segments[i]);
        segments[i] = NULL;
    }



    zone_malloc_fini(&gdata);
    free(base);
    free(segments);


    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif /* DISTRIBUTED */
}