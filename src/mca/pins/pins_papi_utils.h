#ifndef PINS_PAPI_UTILS_H
#define PINS_PAPI_UTILS_H

#define WHICH_CORE_IN_SOCKET 1 // OLD: mostly, just don't choose 0; it interferes with PaRSEC's thread handling
// at this point, any value for WHICH should be acceptable, due to refactoring of PINS finalization code

// this is now in CMAKE config, until dague-hwloc is updated to support dynamic determination
// #define CORES_PER_SOCKET 6 // for ig.icl.utk.edu, an Istanbul Opteron, anyway


#endif
