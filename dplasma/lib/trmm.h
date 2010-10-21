#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "generated/ztrmm_LLN.h"
#include "generated/ztrmm_LLT.h"
#include "generated/ztrmm_LUN.h"
#include "generated/ztrmm_LUT.h"
#include "generated/ztrmm_RLN.h"
#include "generated/ztrmm_RLT.h"
#include "generated/ztrmm_RUN.h"
#include "generated/ztrmm_RUT.h"
#else /* DAGSINGLE */
#include "generated/ctrmm_LLN.h"
#include "generated/ctrmm_LLT.h"
#include "generated/ctrmm_LUN.h"
#include "generated/ctrmm_LUT.h"
#include "generated/ctrmm_RLN.h"
#include "generated/ctrmm_RLT.h"
#include "generated/ctrmm_RUN.h"
#include "generated/ctrmm_RUT.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "generated/dtrmm_LLN.h"
#include "generated/dtrmm_LLT.h"
#include "generated/dtrmm_LUN.h"
#include "generated/dtrmm_LUT.h"
#include "generated/dtrmm_RLN.h"
#include "generated/dtrmm_RLT.h"
#include "generated/dtrmm_RUN.h"
#include "generated/dtrmm_RUT.h"
#else
#include "generated/strmm_LLN.h"
#include "generated/strmm_LLT.h"
#include "generated/strmm_LUN.h"
#include "generated/strmm_LUT.h"
#include "generated/strmm_RLN.h"
#include "generated/strmm_RLT.h"
#include "generated/strmm_RUN.h"
#include "generated/strmm_RUT.h"
#endif
#endif
