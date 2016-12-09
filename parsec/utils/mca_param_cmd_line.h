#ifndef _mca_param_cmd_line_h_
#define _mca_param_cmd_line_h_

#include "parsec_config.h"
#include "parsec/utils/cmd_line.h"

int parsec_mca_cmd_line_setup(parsec_cmd_line_t *cmd);
int parsec_mca_cmd_line_process_args(parsec_cmd_line_t *cmd,
                                    char ***context_env, char ***global_env);

#endif /* _mca_param_cmd_line_h_ */
