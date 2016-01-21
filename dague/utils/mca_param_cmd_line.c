/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "dague_config.h"

#include <stdio.h>
#include <string.h>

#include "dague/utils/cmd_line.h"
#include "dague/utils/argv.h"
#include "dague/utils/mca_param.h"
#include "dague/utils/dague_environ.h"
#include "dague/constants.h"


/*
 * Private variables
 */

/*
 * Private functions
 */
static int process_arg(const char *param, const char *value,
                       char ***params, char ***values);
static void add_to_env(char **params, char **values, char ***env);


/*
 * Add -mca to the possible command line options list
 */
int dague_mca_cmd_line_setup(dague_cmd_line_t *cmd)
{
    int ret = DAGUE_SUCCESS;

    ret = dague_cmd_line_make_opt3(cmd, '\0', "mca", "mca", 2,
                                  "Pass context-specific MCA parameters; they are considered global if --gmca is not used and only one context is specified (arg0 is the parameter name; arg1 is the parameter value)");
    if (DAGUE_SUCCESS != ret) {
        return ret;
    }

    ret = dague_cmd_line_make_opt3(cmd, '\0', "gmca", "gmca", 2,
                                  "Pass global MCA parameters that are applicable to all contexts (arg0 is the parameter name; arg1 is the parameter value)");

    if (DAGUE_SUCCESS != ret) {
        return ret;
    }

    {
        dague_cmd_line_init_t entry =
            {"dague_mca_param_file_prefix", '\0', "am", NULL, 1,
             NULL, DAGUE_CMD_LINE_TYPE_STRING,
             "Aggregate MCA parameter set file list"
            };
        ret = dague_cmd_line_make_opt_mca(cmd, entry);
        if (DAGUE_SUCCESS != ret) {
            return ret;
        }
    }

    return ret;
}


/*
 * Look for and handle any -mca options on the command line
 */
int dague_mca_cmd_line_process_args(dague_cmd_line_t *cmd,
                                    char ***context_env, char ***global_env)
{
  int i, num_insts;
  char **params;
  char **values;

  /* If no relevant parameters were given, just return */

  if (!dague_cmd_line_is_taken(cmd, "mca") &&
      !dague_cmd_line_is_taken(cmd, "gmca")) {
      return DAGUE_SUCCESS;
  }

  /* Handle app context-specific parameters */

  num_insts = dague_cmd_line_get_ninsts(cmd, "mca");
  params = values = NULL;
  for (i = 0; i < num_insts; ++i) {
      process_arg(dague_cmd_line_get_param(cmd, "mca", i, 0),
                  dague_cmd_line_get_param(cmd, "mca", i, 1),
                  &params, &values);
  }
  if (NULL != params) {
      add_to_env(params, values, context_env);
      dague_argv_free(params);
      dague_argv_free(values);
  }

  /* Handle global parameters */

  num_insts = dague_cmd_line_get_ninsts(cmd, "gmca");
  params = values = NULL;
  for (i = 0; i < num_insts; ++i) {
      process_arg(dague_cmd_line_get_param(cmd, "gmca", i, 0),
                  dague_cmd_line_get_param(cmd, "gmca", i, 1),
                  &params, &values);
  }
  if (NULL != params) {
      add_to_env(params, values, global_env);
      dague_argv_free(params);
      dague_argv_free(values);
  }

  /* All done */

  return DAGUE_SUCCESS;
}


/*
 * Process a single MCA argument.
 */
static int process_arg(const char *param, const char *value,
                       char ***params, char ***values)
{
    int i;
    char *new_str;

    /* Look to see if we've already got an -mca argument for the same
       param.  Check against the list of MCA param's that we've
       already saved arguments for. */

    for (i = 0; NULL != *params && NULL != (*params)[i]; ++i) {
        if (0 == strcmp(param, (*params)[i])) {
            asprintf(&new_str, "%s,%s", (*values)[i], value);
            free((*values)[i]);
            (*values)[i] = new_str;

            return DAGUE_SUCCESS;
        }
    }

    /* If we didn't already have a value for the same param, save
       this one away */

    dague_argv_append_nosize(params, param);
    dague_argv_append_nosize(values, value);

    return DAGUE_SUCCESS;
}


static void add_to_env(char **params, char **values, char ***env)
{
    int i;
    char *name;

    /* Loop through all the args that we've gotten and make env
       vars of the form OMPI_MCA_*=value. */

    for (i = 0; NULL != params && NULL != params[i]; ++i) {
        dague_register_mca_param( params[i], values[i], env );
    }
}
