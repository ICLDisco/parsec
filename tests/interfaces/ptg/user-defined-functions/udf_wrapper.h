#ifndef _UDF_WRAPPER_H
#define _UDF_WRAPPER_H

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

typedef enum {
    UDF_TT_NBLOCAL,
    UDF_TT_NBLOCAL_MAKEKEY,
    UDF_TT_NBLOCAL_STARTUP,
    UDF_TT_NBLOCAL_HASHSTRUCT,
    UDF_TT_NBLOCAL_MAKEKEY_STARTUP,
    UDF_TT_MAX
} udf_task_type_t;

typedef int (*udf_logger_fn_t)(int, udf_task_type_t task_type);

extern char *UDF_TASKTYPE_NAME[UDF_TT_MAX];

#include "udf.h"

#endif
