#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "generated/ztrsm_LLN.h"
#include "generated/ztrsm_LLT.h"
#include "generated/ztrsm_LUN.h"
#include "generated/ztrsm_LUT.h"
#include "generated/ztrsm_RLN.h"
#include "generated/ztrsm_RLT.h"
#include "generated/ztrsm_RUN.h"
#include "generated/ztrsm_RUT.h"
#else /* DAGSINGLE */
#include "generated/ctrsm_LLN.h"
#include "generated/ctrsm_LLT.h"
#include "generated/ctrsm_LUN.h"
#include "generated/ctrsm_LUT.h"
#include "generated/ctrsm_RLN.h"
#include "generated/ctrsm_RLT.h"
#include "generated/ctrsm_RUN.h"
#include "generated/ctrsm_RUT.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "generated/dtrsm_LLN.h"
#include "generated/dtrsm_LLT.h"
#include "generated/dtrsm_LUN.h"
#include "generated/dtrsm_LUT.h"
#include "generated/dtrsm_RLN.h"
#include "generated/dtrsm_RLT.h"
#include "generated/dtrsm_RUN.h"
#include "generated/dtrsm_RUT.h"
#else
#include "generated/strsm_LLN.h"
#include "generated/strsm_LLT.h"
#include "generated/strsm_LUN.h"
#include "generated/strsm_LUT.h"
#include "generated/strsm_RLN.h"
#include "generated/strsm_RLT.h"
#include "generated/strsm_RUN.h"
#include "generated/strsm_RUT.h"
#endif
#endif
