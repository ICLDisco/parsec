#ifndef PINS_PAPI_UTILS_H
#define PINS_PAPI_UTILS_H

#ifdef PINS_SHARED_L3_MISSES
#define DO_SOCKET_MEASUREMENTS 1
#else
#define DO_SOCKET_MEASUREMENTS 0
#endif

#define WHICH_CORE_IN_SOCKET 1 //mostly, just don't choose 0; it interferes with PaRSEC's thread handling
#define CORES_PER_SOCKET 6 // for ig.icl.utk.edu, an Istanbul Opteron, anyway

#endif
