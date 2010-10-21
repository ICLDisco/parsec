#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "ztrmm_LLN.h"
#include "ztrmm_LLT.h"
#include "ztrmm_LUN.h"
#include "ztrmm_LUT.h"
#include "ztrmm_RLN.h"
#include "ztrmm_RLT.h"
#include "ztrmm_RUN.h"
#include "ztrmm_RUT.h"
#else /* DAGSINGLE */
#include "ctrmm_LLN.h"
#include "ctrmm_LLT.h"
#include "ctrmm_LUN.h"
#include "ctrmm_LUT.h"
#include "ctrmm_RLN.h"
#include "ctrmm_RLT.h"
#include "ctrmm_RUN.h"
#include "ctrmm_RUT.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "dtrmm_LLN.h"
#include "dtrmm_LLT.h"
#include "dtrmm_LUN.h"
#include "dtrmm_LUT.h"
#include "dtrmm_RLN.h"
#include "dtrmm_RLT.h"
#include "dtrmm_RUN.h"
#include "dtrmm_RUT.h"
#else
#include "strmm_LLN.h"
#include "strmm_LLT.h"
#include "strmm_LUN.h"
#include "strmm_LUT.h"
#include "strmm_RLN.h"
#include "strmm_RLT.h"
#include "strmm_RUN.h"
#include "strmm_RUT.h"
#endif
#endif
