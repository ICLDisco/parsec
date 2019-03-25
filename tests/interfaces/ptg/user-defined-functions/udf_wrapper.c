#include "udf_wrapper.h"
#include "udf.h"

char *UDF_TASKTYPE_NAME[UDF_TT_MAX] =
    { "nb_local_tasks_fn",
      "nb_local_tasks_fn + make_key_fn",
      "nb_local_tasks_fn + startup_fn",
      "nb_local_tasks_fn + make_key_fn + startup_fn",
    };
