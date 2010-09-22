#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "ztrsm_LLN.h"
#include "ztrsm_LLT.h"
#include "ztrsm_LUN.h"
#include "ztrsm_LUT.h"
#include "ztrsm_RLN.h"
#include "ztrsm_RLT.h"
#include "ztrsm_RUN.h"
#include "ztrsm_RUT.h"
#else /* DAGSINGLE */
#include "ctrsm_LLN.h"
#include "ctrsm_LLT.h"
#include "ctrsm_LUN.h"
#include "ctrsm_LUT.h"
#include "ctrsm_RLN.h"
#include "ctrsm_RLT.h"
#include "ctrsm_RUN.h"
#include "ctrsm_RUT.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "dtrsm_LLN.h"
#include "dtrsm_LLT.h"
#include "dtrsm_LUN.h"
#include "dtrsm_LUT.h"
#include "dtrsm_RLN.h"
#include "dtrsm_RLT.h"
#include "dtrsm_RUN.h"
#include "dtrsm_RUT.h"
#else
#include "strsm_LLN.h"
#include "strsm_LLT.h"
#include "strsm_LUN.h"
#include "strsm_LUT.h"
#include "strsm_RLN.h"
#include "strsm_RLT.h"
#include "strsm_RUN.h"
#include "strsm_RUT.h"
#endif
#endif
