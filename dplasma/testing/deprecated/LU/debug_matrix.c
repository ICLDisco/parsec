#include "dague.h"
#include "data_dist/matrix/matrix.h"

#define MAXDBLSTRLEN 16
#define DEBUG_MATRICES 0

#if DEBUG_MATRICES
extern FILE* matout;
static void debug_warning(int core, const char *when, const char *function, int p1, int p2, int p3)
{
    int m, n, len, pos;
    double *a, *l;
    int *ipiv;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    if( p2 != -1 ) {
        if( p3 != -1 ) {
            fprintf(matout, "[%02d/%02d] %s call of %s(%02d, %02d, %02d)  ", rank, core, when, function, p1, p2, p3);
        } else {
            fprintf(matout, "[%02d/%02d] %s call of %s(%02d, %02d)  ", rank, core, when, function, p1, p2);
        }
    } else {
        fprintf(matout, "[%02d/%02d] %s call of %s(%02d)  ", rank, core, when, function, p1, p2);
    }
}

static void debug_tile(tiled_matrix_desc_t* ddesc, int core, double *a, char *name, int tilem, int tilen)
{
    int m, n, len, pos;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * ddesc->nb;
    line = (char *)malloc( len );

    fprintf(matout, "[%02d/%02d] %s(%02d, %02d) = \r", rank, core, name, tilem, tilen);
    pos = 0;
    for(m = 0; m < ddesc->mb; m++) {
        for(n = 0; n < ddesc->nb; n++) {
            pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + ddesc->mb * n]);
        }
        fprintf(matout, "[%02d/%02d]   %s\r", rank, core, line);
        pos = 0;
    }
    fflush(matout);
    free(line);
}

static void debug_lower(tiled_matrix_desc_t* ddesc, int core, double *a, char *name, int tilem, int tilen)
{
    int m, n, len, pos;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * ddesc->nb;
    line = (char *)malloc( len );

    fprintf(matout, "[%02d/%02d] %s(%02d, %02d) = \r", rank, core, name, tilem, tilen);
    pos = 0;
    for(m = 0; m < ddesc->mb; m++) {
        for(n = 0; n < ddesc->nb; n++) {
            if( m <= n )
                pos += snprintf(line + pos, len-pos, "%9.5f ", 0.0);
            else
                pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + ddesc->mb * n]);
        }
        fprintf(matout, "[%02d/%02d]   %s\r", rank, core, line);
        pos = 0;
    }
    fflush(matout);
    free(line);
}

static void debug_upper(tiled_matrix_desc_t* ddesc, int core, double *a, char *name, int tilem, int tilen)
{
    int m, n, len, pos;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * ddesc->nb;
    line = (char *)malloc( len );

    fprintf(matout, "[%02d/%02d] %s(%02d, %02d) = \r", rank, core, name, tilem, tilen);
    pos = 0;
    for(m = 0; m < ddesc->mb; m++) {
        for(n = 0; n < ddesc->nb; n++) {
            if( m <= n )
                pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + ddesc->mb * n]);
            else
                pos += snprintf(line + pos, len-pos, "%9.5f ", 0.0);
        }
        fprintf(matout, "[%02d/%02d]   %s\r", rank, core, line);
        pos = 0;
    }
    fflush(matout);
    free(line);
}

static void debug_l(tiled_matrix_desc_t* ddesc, int core, double* a, char* name, int tilen, int tilem)
{
    int m, n, len, pos;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * ddesc->nb;
    line = (char *)malloc( len );

    fprintf(matout, "[%02d/%02d] %s(%02d, %02d) = \r", rank, core, name, tilem, tilen);
    pos = 0;
    for(m = 0; m < (ddesc->mb-1); m++) {
        for(n = 0; n < ddesc->nb; n++) {
            pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + ddesc->mb * n]);
        }
        fprintf(matout, "[%02d/%02d]   %s\r", rank, core, line);
        pos = 0;
    }
    fflush(matout);
    free(line);
}



static void debug_ipiv(tiled_matrix_desc_t* ddesc, int core, int *a, char *name, int tilen, int tilem)
{
    int m, len, pos;
    char *line;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * ddesc->nb;
    line = (char *)malloc( len );

    fprintf(matout, "[%02d/%02d] %s(%02d, %02d) = ", rank, core, name, tilem, tilen);
    pos = 0;
    for(m = 0; m < ddesc->nb; m++) {
        pos += snprintf(line + pos, len-pos, "%3d ", a[m]);
    }
    fprintf(matout, "[%02d/%02d]   %s\n", rank, core, line);
    fflush(matout);
    free(line);
}
#else
#define debug_ipiv(ddesc, core, a, name, tilen, tilem)
#define debug_l(ddesc, core, a, name, tilen, tilem)
#define debug_tile(ddesc, core, a, name, tilen, tilem)
#define debug_upper(ddesc, core, a, name, tilen, tilem)
#define debug_lower(ddesc, core, a, name, tilen, tilem)
#define debug_warning(core, when, function, p1, p2, p3)
#endif
