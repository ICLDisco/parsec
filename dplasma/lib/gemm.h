#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "zgemm_NN.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"
#else /* DAGSINGLE */
#include "cgemm_NN.h"
#include "cgemm_NT.h"
#include "cgemm_TN.h"
#include "cgemm_TT.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "dgemm_NN.h"
#include "dgemm_NT.h"
#include "dgemm_TN.h"
#include "dgemm_TT.h"
#else
#include "sgemm_NN.h"
#include "sgemm_NT.h"
#include "sgemm_TN.h"
#include "sgemm_TT.h"
#endif
#endif
