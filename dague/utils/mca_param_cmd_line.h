#ifndef _mca_param_cmd_line_h_
#define _mca_param_cmd_line_h_

#include "dague_config.h"
#include "dague/utils/cmd_line.h"

int dague_mca_cmd_line_setup(dague_cmd_line_t *cmd);
int dague_mca_cmd_line_process_args(dague_cmd_line_t *cmd,
                                    char ***context_env, char ***global_env);

#endif /* _mca_param_cmd_line_h_ */
