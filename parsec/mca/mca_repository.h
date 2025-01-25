#ifndef MCA_REPOSITORY_H
#define MCA_REPOSITORY_H
/*
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"

void mca_components_repository_init(void);

int mca_components_belongs_to_user_list(char **list, const char *name);
char **mca_components_get_user_selection(char *type);
void mca_components_free_user_list(char **list);

char *mca_components_list_compiled(char* type_name);
mca_base_component_t **mca_components_open_bytype(char *type);
mca_base_component_t *mca_component_open_byname(char *type, char *name);

/**
 * @brief Queries which component in a list of opened components can be
 *   used, and returns the corresponding module
 *
 * @param opened_components a list of opened components, are returned by
 *   mca_components_open_bytype
 * @param selected_module the module corresponding to the selected component
 * @param selected_component the selected component (which is removed from
 *   the opened_components list)
 */
void mca_components_query(mca_base_component_t **opened_components,
                          mca_base_module_t **selected_module,
                          mca_base_component_t **selected_component);

/**
 * @brief Queries a single component opened by name
 *
 * @details It is sometimes useful to query a single component, that has been selected
 *   explicitly via mca_component_open_byname (e.g. specific taskpools can only work
 *   with the user_trigger termination detector, so only this component must be selected).
 *   This function queries if the component can indeed by used (and it's probably an error
 *   if it cannot), and returns the corresponding module.
 *
 * @param opened_component the opened component returned by mca_component_open_byname
 * @return mca_base_module_t* the module corresponding to this component, or NULL if
 *   that component cannot be selected.
 */
mca_base_module_t *mca_component_query(mca_base_component_t *opened_component);

void mca_component_close(mca_base_component_t *opened_component);
void mca_components_close(mca_base_component_t **opened_components);

#endif /* MCA_REPOSITORY_H */
