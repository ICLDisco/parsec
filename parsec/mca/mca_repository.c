#include "parsec_config.h"
#include "parsec/debug.h"
#include "parsec/mca/mca.h"
#include "parsec/utils/mca_param.h"
#include <string.h>
#include <stdlib.h>
#include "parsec/mca/mca_repository.h"

#define MCA_REPOSITORY_C
#include "parsec/mca/mca_static_components.h"

void mca_components_repository_init(void)
{
    mca_static_components_init();
}

int mca_components_belongs_to_user_list(char **list, const char *name)
{
    int i;
    if( list == NULL ) return 1;
    for(i = 0; list[i] != NULL; i++)
        if( strcmp(list[i], name) == 0 )
            return 1;
    return 0;
}

char **mca_components_get_user_selection(char *type)
{
    char *param, **list;
    int idx, nb, i, n;

    idx = parsec_mca_param_find("mca", NULL, type);
    if( idx == PARSEC_ERROR )
        return NULL;

    parsec_mca_param_lookup_string(idx, &param);
    if( param == NULL )
        return NULL;

    for(nb = 1, i = 0; param[i] != '\0'; i++) {
        if( param[i] == ',' )
            nb++;
    }
    list = (char**)malloc( (nb+1)*sizeof(char*) );
    for(nb = 0, i = 0, n = 0; param[i] != '\0'; i++) {
        if( param[i] == ',' ) {
            list[nb] = (char*)malloc(i-n+1);
            memcpy( list[nb], &param[n], i-n );
            list[nb][i-n] = '\0';
            nb++;
            n = i+1;
        }
    }
    list[nb] = (char*)malloc(i-n+1);
    memcpy( list[nb], &param[n], i-n );
    list[nb][i-n] = '\0';
    nb++;
    list[nb] = NULL;

    return list;
}

void mca_components_free_user_list(char **list)
{
    int i;
    if( NULL != list ) {
        for(i = 0; list[i] != NULL; i++) {
            free(list[i]);
        }
        free(list);
    }
}

mca_base_component_t **mca_components_open_bytype(char *type)
{
    int i, nb, n;
    mca_base_component_t **opened_components;
    char **list;

    list = mca_components_get_user_selection(type);

    nb = 0;
    for(i = 0; mca_static_components[i] != NULL; i++) {
        if( !strcmp( mca_static_components[i]->mca_type_name, type ) &&
            mca_components_belongs_to_user_list(list, mca_static_components[i]->mca_component_name) )
            nb++;
    }
    opened_components = (mca_base_component_t**)malloc(sizeof(mca_base_component_t*) * (nb+1));
    n = 0;
    for(i = 0; (n < nb) && (mca_static_components[i] != NULL); i++) {
        if( !strcmp( mca_static_components[i]->mca_type_name, type ) &&
            mca_components_belongs_to_user_list(list, mca_static_components[i]->mca_component_name) ) {
            if( ( (NULL != mca_static_components[i]->mca_open_component) &&
                  (mca_static_components[i]->mca_open_component()) ) ||
                ( NULL ==  mca_static_components[i]->mca_open_component ) ) {
                opened_components[n] = mca_static_components[i];
                if( NULL != mca_static_components[i]->mca_register_component_params ) {
                    mca_static_components[i]->mca_register_component_params();
                }
                n++;
            }
        }
    }

    mca_components_free_user_list(list);

    opened_components[n] = NULL;
    return opened_components;
}

void mca_components_query(mca_base_component_t **opened_components,
                          mca_base_module_t **selected_module,
                          mca_base_component_t **selected_component)
{
    int i, s = -1;
    int priority = -1, p;
    mca_base_module_t *m = NULL;

    for(i = 0; opened_components[i] != NULL; i++) {
        if( opened_components[i]->mca_query_component != NULL ) {
            opened_components[i]->mca_query_component(&m, &p);
            if( p > priority ) {
                *selected_module = m;
                s = i;
                priority = p;
                *selected_component = opened_components[i];
            }
        }
    }
    /* Remove the selected component from the opened list */
    if( s != -1 ) {
        opened_components[s] = opened_components[i-1];
        opened_components[i-1] = NULL;
    }
}

void mca_component_close(mca_base_component_t *opened_component)
{
    if( opened_component->mca_close_component != NULL ) {
        opened_component->mca_close_component();
    }
}

void mca_components_close(mca_base_component_t **opened_components)
{
    int i;

    for(i = 0; opened_components[i] != NULL; i++) {
        mca_component_close( opened_components[i] );
        opened_components[i] = NULL;
    }
    free( opened_components );
}
