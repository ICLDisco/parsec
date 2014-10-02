#ifndef MCA_REPOSITORY_H
#define MCA_REPOSITORY_H

#include "dague_config.h"
#include "dague/mca/mca.h"

void mca_components_repository_init(void);

int mca_components_belongs_to_user_list(char **list, const char *name);
char **mca_components_get_user_selection(char *type);
void mca_components_free_user_list(char **list);

mca_base_component_t **mca_components_open_bytype(char *type);
void mca_components_query(mca_base_component_t **opened_components,
                          mca_base_module_t **selected_module,
                          mca_base_component_t **selected_component);
void mca_component_close(mca_base_component_t *opened_component);
void mca_components_close(mca_base_component_t **opened_components);

#endif /* MCA_REPOSITORY_H */
