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

#include "parsec/parsec_config.h"

#include <stdio.h>
#include <string.h>

#include "parsec/utils/cmd_line.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/parsec_environ.h"
#include "parsec/constants.h"


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
int parsec_mca_cmd_line_setup(parsec_cmd_line_t *cmd)
{
    int ret = PARSEC_SUCCESS;

    ret = parsec_cmd_line_make_opt3(cmd, '\0', "mca", "mca", 2,
                                  "Pass context-specific MCA parameters; they are considered global if --gmca is not used and only one context is specified (arg0 is the parameter name; arg1 is the parameter value)");
    if (PARSEC_SUCCESS != ret) {
        return ret;
    }

    ret = parsec_cmd_line_make_opt3(cmd, '\0', "gmca", "gmca", 2,
                                  "Pass global MCA parameters that are applicable to all contexts (arg0 is the parameter name; arg1 is the parameter value)");

    if (PARSEC_SUCCESS != ret) {
        return ret;
    }

    {
        parsec_cmd_line_init_t entry =
            {"parsec_mca_param_file_prefix", '\0', "am", NULL, 1,
             NULL, PARSEC_CMD_LINE_TYPE_STRING,
             "Aggregate MCA parameter set file list"
            };
        ret = parsec_cmd_line_make_opt_mca(cmd, entry);
        if (PARSEC_SUCCESS != ret) {
            return ret;
        }
    }

    return ret;
}


/*
 * Look for and handle any -mca options on the command line
 */
int parsec_mca_cmd_line_process_args(parsec_cmd_line_t *cmd,
                                    char ***context_env, char ***global_env)
{
  int i, num_insts;
  char **params;
  char **values;

  /* If no relevant parameters were given, just return */

  if (!parsec_cmd_line_is_taken(cmd, "mca") &&
      !parsec_cmd_line_is_taken(cmd, "gmca")) {
      return PARSEC_SUCCESS;
  }

  /* Handle app context-specific parameters */

  num_insts = parsec_cmd_line_get_ninsts(cmd, "mca");
  params = values = NULL;
  for (i = 0; i < num_insts; ++i) {
      process_arg(parsec_cmd_line_get_param(cmd, "mca", i, 0),
                  parsec_cmd_line_get_param(cmd, "mca", i, 1),
                  &params, &values);
  }
  if (NULL != params) {
      add_to_env(params, values, context_env);
      parsec_argv_free(params);
      parsec_argv_free(values);
  }

  /* Handle global parameters */

  num_insts = parsec_cmd_line_get_ninsts(cmd, "gmca");
  params = values = NULL;
  for (i = 0; i < num_insts; ++i) {
      process_arg(parsec_cmd_line_get_param(cmd, "gmca", i, 0),
                  parsec_cmd_line_get_param(cmd, "gmca", i, 1),
                  &params, &values);
  }
  if (NULL != params) {
      add_to_env(params, values, global_env);
      parsec_argv_free(params);
      parsec_argv_free(values);
  }

  /* All done */

  return PARSEC_SUCCESS;
}


/*
 * Process a single MCA argument.
 */
static int process_arg(const char *param, const char *value,
                       char ***params, char ***values)
{
    int i, rc;
    char *new_str;

    /* Look to see if we've already got an -mca argument for the same
       param.  Check against the list of MCA param's that we've
       already saved arguments for. */

    for (i = 0; NULL != *params && NULL != (*params)[i]; ++i) {
        if (0 == strcmp(param, (*params)[i])) {
            rc = asprintf(&new_str, "%s,%s", (*values)[i], value);
            free((*values)[i]);
            (*values)[i] = new_str;

            return PARSEC_SUCCESS;
        }
    }

    /* If we didn't already have a value for the same param, save
       this one away */

    parsec_argv_append_nosize(params, param);
    parsec_argv_append_nosize(values, value);

    (void)rc;
    return PARSEC_SUCCESS;
}


static void add_to_env(char **params, char **values, char ***env)
{
    int i;

    /* Loop through all the args that we've gotten and make env
       vars of the form OMPI_MCA_*=value. */

    for (i = 0; NULL != params && NULL != params[i]; ++i) {
        parsec_setenv_mca_param( params[i], values[i], env );
    }
}
