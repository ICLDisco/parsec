//##################################################################################################

#include <stdio.h>    // Standard stuff
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>
#include <string.h>

#include <fcntl.h>    // Huge TLB pages
#include <sys/mman.h>

#include <math.h>     // Math and MKL
#include <mkl_cblas.h>

#define __USE_UNIX98  // Pthreads
#include <pthread.h>

#include <sys/time.h> // Time

#define MAX_NB 1024
#define MAX_BB 128
#define MAX_THREADS 256
#define HUGE_PAGE_SIZE 2048*1024

#define BSIZE 5
typedef float block_t[BSIZE][BSIZE];
typedef block_t **matrix_t;

#define LL
//#define RL

#define CHECK
//#define LOG

////////////////////////////////////////////////////////////////////////////////////////////////////

struct {
    int cores_num;
    float *A;
    int NB;
    int BB;
} core_in_all;

double GFLOPS;

char Left = 'L', Transpose = 'T', Forward = 'F', Columnwise = 'C', Upper = 'U', Lower = 'L';

void slarnv(int*, int*, int*, float*);
void dump_trace(void);
void diff_matrix(float *A, float *B, int NB, int BB);
void spotrf(char*, int*, float*, int*, int*);

void tile_ssyrk(float*, float*, int);
void tile_spotrf(float *, int);
void tile_sgemm(float *, float *, float *, int);
void tile_strsm(float *, float *, int);
////////////////////////////////////////////////////////////////////////////////////////////////////

double get_current_time(void)
{
    struct timeval  time_val;
    struct timezone time_zone;

    gettimeofday(&time_val, &time_zone);

    return (double)(time_val.tv_sec) + (double)(time_val.tv_usec) / 1000000.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//#define MAX_EVENTS 16384
#define MAX_EVENTS 1048576

int event_num;
int log_events = 1;
double event_start_time [MAX_THREADS]   __attribute__ ((aligned (128)));
double event_end_time   [MAX_THREADS]   __attribute__ ((aligned (128)));
double event_log        [MAX_EVENTS]    __attribute__ ((aligned (128)));

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

#define event_start()\
    pthread_mutex_lock(&log_mutex);\
    event_start_time[my_thread_id()] = get_current_time();\
    pthread_mutex_unlock(&log_mutex);

#define event_end()\
    pthread_mutex_lock(&log_mutex);\
    event_end_time[my_thread_id()] = get_current_time();\
    pthread_mutex_unlock(&log_mutex);

#define log_event(event)\
    pthread_mutex_lock(&log_mutex);\
    event_log[event_num+0] = my_thread_id();\
    event_log[event_num+1] = event_start_time[my_thread_id()];\
    event_log[event_num+2] = event_end_time[my_thread_id()];\
    event_log[event_num+3] = (event);\
    event_num += log_events << 2;\
    event_num &= MAX_EVENTS-1;\
    pthread_mutex_unlock(&log_mutex);

int my_thread_id()
{
    static int num_threads = 0;
    static pthread_t threads[MAX_THREADS];
    pthread_t self;
    int i;

    self = pthread_self();
    for (i = 0; i < num_threads; i++)
      if (pthread_equal(self, threads[i]))
        return (i);

    threads[num_threads++] = self;
    return (num_threads);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv)
{
    assert(argc == 2);
    int BB = atoi(argv[1]); assert(BB <= MAX_BB);
    int N = BB*BSIZE;
    int NxN = N*N;

    double start_time, end_time, elapsed_time;
    char mem_file_name[32];
    int  huge_size;
    int  fmem;
    char *mem_block = 0;
    int m, n, k;
    int i, j;
    int step;
    int X, Y, x, y;
    matrix_t a;
    int INFO;
    int ONE = 1;
    int ISEED[4] = {0,0,0,1};

    float *A1   = memalign(128,  NxN*sizeof(float));
    float *A2   = memalign(128,  NxN*sizeof(float));

    a = (matrix_t)malloc(BB * sizeof(block_t*));
    for (n = 0; n < BB; n++){
      a[n] = (block_t*)mem_block; mem_block += (BB * sizeof(block_t));}

    huge_size = (unsigned long)mem_block;
    huge_size = (huge_size + HUGE_PAGE_SIZE-1) & ~(HUGE_PAGE_SIZE-1);
    sprintf(mem_file_name, "/huge/huge_tlb_page.bin");
    assert((fmem = open(mem_file_name, O_CREAT | O_RDWR, 0755)) != -1);
    remove(mem_file_name);
    mem_block = (char*)mmap(0, huge_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fmem, 0);
    assert(mem_block != MAP_FAILED);

    //for (i = 0; i < huge_size / sizeof(float); i++)
        //((float*)mem_block)[i] = 0.5 - (float)rand() / RAND_MAX;

    for (n = 0; n < BB; n++){
      a[n] = (block_t*)(mem_block + (unsigned long)(a[n]));}

    slarnv(&ONE, ISEED, &NxN, A1);
    // Make the matrix SYM
    for (i=0;i<N;i++) {
        for (j=i;j<N;j++) {
            *(A1+j*N+i)=*(A1+i*N+j);
	    if (i==j)
               *(A1+j*N+i)=*(A1+i*N+j)+100*N;
        }
    }

    // Move from F77 to BDL
    for (X = 0; X < BB; X++)
      for (Y = 0; Y < BB; Y++)
        for (x = 0; x < BSIZE; x++)
          for (y = 0; y < BSIZE; y++)
              a[Y][X][x][y] = A1[Y*BSIZE + X*BSIZE*N + y + x*N];

#if defined CHECK
    memcpy(A2, A1, NxN*sizeof(float));

    {
    int INFO;
    spotrf(&Lower, &N, A1, &N, &INFO);
    assert(INFO == 0);
    }
#endif

    /*for (i=0;i<N;i++) {
        for (j=0;j<N;j++) {
            printf("%f ", *(A1+j*N+i));
        }
	printf("\n");
    }*/


    start_time = get_current_time();
#if defined LL

    #pragma css start
    // Left-looking tile Cholesky
    for (step = 0; step < BB; step++)
    {
        for (n = 0; n < step; n++)
            tile_ssyrk((float*)(a[step][n]), (float*)(a[step][step]), BSIZE);
        tile_spotrf((float*)(a[step][step]), BSIZE);
        for (m = step+1; m < BB; m++)
        {
            for (n = 0; n < step; n++)
                tile_sgemm((float*)(a[step][n]), (float*)(a[m][n]), (float*)(a[m][step]), BSIZE);
            tile_strsm((float*)(a[step][step]), (float*)(a[m][step]), BSIZE);
        }
	
    }
    #pragma css finish

#else 

    #pragma css start
    // Right-looking tile Cholesky
    for (step = 0; step < BB; step++)
    {
        tile_spotrf((float*)(a[step][step]), BSIZE);
        for (n = step+1; n < BB; n++){
            tile_strsm((float*)(a[step][step]), (float*)(a[n][step]), BSIZE);
            tile_ssyrk((float*)(a[n][step]), (float*)(a[n][n]), BSIZE);
	}
        for (m = step+2; m < BB; m++)
            for (n = step+1; n < m; n++)
                tile_sgemm((float*)(a[n][step]), (float*)(a[m][step]), (float*)(a[m][n]), BSIZE);
    }
    #pragma css finish

#endif

    // Solve the linear system
    for (step = 0; step < BB; step++)
    {
        tile_spotrf((float*)(a[step][step]), BSIZE);
        for (n = step+1; n < BB; n++){
            tile_strsm((float*)(a[step][step]), (float*)(a[n][step]), BSIZE);
            tile_ssyrk((float*)(a[n][step]), (float*)(a[n][n]), BSIZE);
	}
        for (m = step+2; m < BB; m++)
            for (n = step+1; n < m; n++)
                tile_sgemm((float*)(a[n][step]), (float*)(a[m][step]), (float*)(a[m][n]), BSIZE);
    }




    end_time = get_current_time();
    elapsed_time = end_time - start_time;
    GFLOPS = 1.0*N*N*N/3.0 / elapsed_time / 1000000000;

#if defined CHECK
    // Move from BDL to F77
    {int X, Y, x, y;
    for (X = 0; X < BB; X++)
      for (Y = 0; Y < BB; Y++)
        for (x = 0; x < BSIZE; x++)
          for (y = 0; y < BSIZE; y++)
            A2[Y*BSIZE + X*BSIZE*N + y + x*N] = a[Y][X][x][y];}
    diff_matrix(A1, A2, BSIZE, BB);
#endif

#if defined LOG
    dump_trace();
#endif
    printf("\t%.2lf\t%.2lf\n", GFLOPS, GFLOPS / (2.393895 * 8 * 16) * 100.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma css task input(NB) input(A[NB][NB]) inout(T[NB][NB])
void tile_ssyrk(float *A, float *T, int NB)
{
#if defined LOG
       event_start();
#endif

    cblas_ssyrk(
        CblasColMajor,
        CblasLower, CblasNoTrans,
        NB, NB,
       -1.0, A, NB,
        1.0, T, NB);

#if defined LOG
       event_end();
       log_event(0xB060D0); 
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma css task highpriority input(NB) inout(A[NB][NB])
void tile_spotrf(float *A, int NB)
{
    int INFO;

#if defined LOG
       event_start();
#endif

    spotrf("L", &NB, A, &NB, &INFO);

#if defined LOG
       event_end();
       log_event(0x006680); 
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma css task input(NB) input(A[NB][NB], B[NB][NB]) inout(C[NB][NB])
void tile_sgemm(float *A, float *B, float *C, int NB)
{
#if defined LOG
       event_start();
#endif

    cblas_sgemm(
        CblasColMajor,
        CblasNoTrans, CblasTrans,
        NB, NB, NB,
       -1.0, B, NB,
             A, NB,
        1.0, C, NB);

#if defined LOG
       event_end();
       log_event(0xD0F040); 
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma css task input(NB) input(T[NB][NB]) inout(B[NB][NB])
void tile_strsm(float *T, float *B, int NB)
{
#if defined LOG
       event_start();
#endif

    cblas_strsm(
        CblasColMajor,
        CblasRight, CblasLower,
        CblasTrans, CblasNonUnit,
        NB, NB,
        1.0, T, NB,
             B, NB);

#if defined LOG
       event_end();
       log_event(0x00C0F0); 
#endif
}

//##################################################################################################

void dump_trace(void)
{
    char trace_file_name[32];
    FILE *trace_file;
    int event;
    double scale = 30000.0;


    sprintf(trace_file_name, "trace_%d.svg", (int)(time(NULL)));
    trace_file = fopen(trace_file_name, "w");
    assert(trace_file != NULL);

    fprintf(trace_file,
        "<svg width=\"200mm\" height=\"40mm\" viewBox=\"0 0 20000 4000\">\n"
        "  <g>\n");

    for (event = 4; event < event_num; event += 4)
    {
        int    thread = event_log[event+0];
        double start  = event_log[event+1];
        double end    = event_log[event+2];
        int    color  = event_log[event+3];

        start -= event_log[2];
        end   -= event_log[2];

        fprintf(trace_file,
            "    "
            "<rect x=\"%.2lf\" y=\"%.0lf\" width=\"%.2lf\" height=\"%.0lf\" "
            "fill=\"#%06x\" stroke=\"#000000\" stroke-width=\"1\"/>\n",
            start * scale,
            thread * 100.0,
            (end - start) * scale,
            90.0,
            color);
    }

    fprintf(trace_file,
        "  </g>\n"
        "</svg>\n");

    fclose(trace_file);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

void diff_matrix(float *A, float *B, int NB, int BB)
{
    int X, Y, x, y, i;
    int N = NB*BB;

    printf("\n");
    for (Y = 0; Y < BB; Y++) {
      for (y = 0; y < NB; y++) {
        for (X = 0; X < BB; X++) {
          for (x = 0; x < NB; x++) {

            float a, b, c, d, e;
            a = fabs(A[(Y*NB+y) + (X*NB+x)*N]);
            b = fabs(B[(Y*NB+y) + (X*NB+x)*N]);
            c = max(a, b);
            d = min(a, b);
            e = (c - d) / d;

            printf("%c", e < 0.001 ? '.' : '#');
            //if (x == 3) x = NB-5;
            //if (x == 7) x = NB-1;
          }
          printf("  |");
        }
        printf("\n");
        //if (y == 3) y = NB-5;
        //if (y == 7) y = NB-1;
      }
      if (Y < BB-1)
        for (i = 0; i < BB*12; i++) printf("=");
      printf("\n");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

