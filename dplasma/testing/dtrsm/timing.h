#ifndef TIMING_H
#define TIMING_H

static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(print) do { \
  TIME_STOP(); \
  printf("[%d] TIMED %f s :\t", rank, time_elapsed); \
  printf print; \
} while(0)


#ifdef USE_MPI
# define SYNC_TIME_START() do {                 \
        MPI_Barrier(MPI_COMM_WORLD);            \
        sync_time_elapsed = get_cur_time();     \
    } while(0)
# define SYNC_TIME_STOP() do {                                  \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
    } while(0)
# define SYNC_TIME_PRINT(print) do {                                \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("### TIMED %f s :\t", sync_time_elapsed);    \
            printf print;                                           \
        }                                                           \
  } while(0)

/* overload exit in MPI mode */
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)

#else 
# define SYNC_TIME_START() do { sync_time_elapsed = get_cur_time(); } while(0)
# define SYNC_TIME_STOP() do { sync_time_elapsed = get_cur_time() - sync_time_elapsed; } while(0)
# define SYNC_TIME_PRINT(print) do {                                \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("### TIMED %f doing\t", sync_time_elapsed);  \
            printf print;                                           \
        }                                                           \
    } while(0)
#endif

#endif /* TIMING_H */
