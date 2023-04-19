#ifndef _mca_param_cmd_line_h_
#define _mca_param_cmd_line_h_

#include "parsec/utils/cmd_line.h"

BEGIN_C_DECLS

int parsec_mca_cmd_line_setup(parsec_cmd_line_t *cmd);
int parsec_mca_cmd_line_process_args(parsec_cmd_line_t *cmd,
                                    char ***context_env, char ***global_env);

END_C_DECLS

#endif /* _mca_param_cmd_line_h_ */
