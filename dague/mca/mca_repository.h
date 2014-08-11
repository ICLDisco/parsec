#ifndef MCA_REPOSITORY_H
#define MCA_REPOSITORY_H

#include "dague_config.h"
#include "dague/mca/mca.h"

void mca_components_repository_init(void);

mca_base_component_t **mca_components_open_bytype(char *type);
void mca_components_query(mca_base_component_t **opened_components,
                          mca_base_module_t **selected_module,
                          mca_base_component_t **selected_component);
void mca_component_close(mca_base_component_t *opened_component);
void mca_components_close(mca_base_component_t **opened_components);

#endif /* MCA_REPOSITORY_H */
