/*
 * Copyright (c) 2004-2008 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2011 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "parsec/parsec_config.h"
#include <parsec/constants.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef PARSEC_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef PARSEC_HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#ifdef PARSEC_HAVE_STDBOOL_H
#include <stdbool.h>
#endif
#ifdef PARSEC_HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "parsec/class/list_item.h"
#include "parsec/class/list.h"
#include <parsec/class/parsec_value_array.h>
#include <parsec/utils/mca_param.h>
#include <parsec/utils/mca_param_internal.h>
#include <parsec/utils/installdirs.h>
#include <parsec/utils/output.h>
#include <parsec/utils/os_path.h>
#include <parsec/utils/argv.h>
#include <parsec/utils/show_help.h>
#include <parsec/utils/parsec_environ.h>
#include <parsec/utils/keyval_parse.h>

/*
 * Local types
 */

typedef struct {
    /* Base class */
    parsec_list_item_t super;

    /* String of the type name or NULL */
    char *si_type_name;
    /* String of the component name */
    char *si_component_name;
    /* String of the param name */
    char *si_param_name;
    /* Full name of the synonym */
    char *si_full_name;
    /* Name of the synonym's corresponding environment variable */
    char *si_env_var_name;

    /* Whether this synonym is a deprecated name or not */
    bool si_deprecated;
    /* Whether we've shown a warning that this synonym has been
       displayed or not */
    bool si_deprecated_warning_shown;
} parsec_syn_info_t;

/*
 * Public variables
 *
 * This variable is public, but not advertised in mca_param.h.
 * It's only public so that the file parser can see it.
 */
parsec_list_t parsec_mca_param_file_values;

/*
 * local variables
 */
static parsec_value_array_t mca_params;
static const char *mca_prefix = "PARSEC_MCA_";
static char *home = NULL;
static bool initialized = false;

/*
 * local functions
 */
#if defined(__WINDOWS__)
static int read_keys_from_registry(HKEY hKey, char *sub_key, char *current_name);
#endif  /* defined(__WINDOWS__) */
static int read_files(char *file_list);
static int param_register(const char *type_name,
                          const char *component_name,
                          const char *param_name,
                          const char *help_msg,
                          parsec_mca_param_type_t type,
                          bool internal,
                          bool read_only,
                          parsec_mca_param_storage_t *default_value,
                          parsec_mca_param_storage_t *file_value,
                          parsec_mca_param_storage_t *override_value,
                          parsec_mca_param_storage_t *current_value);
static int syn_register(int index_orig, const char *syn_type_name,
                        const char *syn_component_name,
                        const char *syn_param_name, bool deprecated);
static bool param_lookup(size_t index, parsec_mca_param_storage_t *storage,
                         parsec_mca_param_source_t *source,
                         char **source_file);
static bool param_set_override(size_t index,
                               parsec_mca_param_storage_t *storage,
                               parsec_mca_param_type_t type);
static bool lookup_override(parsec_mca_param_t *param,
                            parsec_mca_param_storage_t *storage);
static bool lookup_env(parsec_mca_param_t *param,
                       parsec_mca_param_storage_t *storage);
static bool lookup_file(parsec_mca_param_t *param,
                        parsec_mca_param_storage_t *storage,
                        char **source_file);
static bool lookup_default(parsec_mca_param_t *param,
                           parsec_mca_param_storage_t *storage);
static bool set(parsec_mca_param_type_t type,
                parsec_mca_param_storage_t *dest, parsec_mca_param_storage_t *src);
static void param_constructor(parsec_mca_param_t *p);
static void param_destructor(parsec_mca_param_t *p);
static void fv_constructor(parsec_mca_param_file_value_t *p);
static void fv_destructor(parsec_mca_param_file_value_t *p);
static void info_constructor(parsec_mca_param_info_t *p);
static void info_destructor(parsec_mca_param_info_t *p);
static void syn_info_constructor(parsec_syn_info_t *si);
static void syn_info_destructor(parsec_syn_info_t *si);
static parsec_mca_param_type_t param_type_from_index (size_t index);

/*
 * Make the class instance for parsec_mca_param_t
 */
OBJ_CLASS_INSTANCE(parsec_mca_param_t, parsec_object_t,
                   param_constructor, param_destructor);
OBJ_CLASS_INSTANCE(parsec_mca_param_file_value_t, parsec_list_item_t,
                   fv_constructor, fv_destructor);
OBJ_CLASS_INSTANCE(parsec_mca_param_info_t, parsec_list_item_t,
                   info_constructor, info_destructor);
OBJ_CLASS_INSTANCE(parsec_syn_info_t, parsec_list_item_t,
                   syn_info_constructor, syn_info_destructor);

/*
 * Set it up
 */
int parsec_mca_param_init(void)
{
    if (!initialized) {

        /* Init the value array for the param storage */

        OBJ_CONSTRUCT(&mca_params, parsec_value_array_t);
        parsec_value_array_init(&mca_params, sizeof(parsec_mca_param_t));

        /* Init the file param value list */

        OBJ_CONSTRUCT(&parsec_mca_param_file_values, parsec_list_t);

        /* Set this before we register the parameter, below */

        initialized = true;

        parsec_mca_param_recache_files();
    }

    return PARSEC_SUCCESS;
}

int parsec_mca_param_recache_files(void)
{
    int rc;
    char *files, *new_files = NULL;

    /* We may need this later */
    home = (char*)parsec_home_directory();

#if defined(PARSEC_WANT_HOME_CONFIG_FILES)
    rc = asprintf(&files,
                  "%s"PARSEC_PATH_SEP".parsec"PARSEC_PATH_SEP"mca-params.conf%c%s"PARSEC_PATH_SEP"parsec-mca-params.conf",
                  home, PARSEC_ENV_SEP, parsec_install_dirs.sysconfdir);
    assert(0 < rc);
#else
    rc = asprintf(&files,
                  "%s"PARSEC_PATH_SEP"parsec-mca-params.conf",
                  parsec_install_dirs.sysconfdir);
    assert(0 < rc);
#endif  /* defined(PARSEC_WANT_HOME_CONFIG_FILES) */
    if (-1 == rc) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    /* Initialize a parameter that says where MCA param files can
       be found */

    (void)parsec_mca_param_reg_string_name("mca", "param_files",
                                          "Path for MCA configuration files containing default parameter values",
                                          false, false, files, &new_files);

    read_files(new_files);
#if defined(__WINDOWS__)
    read_keys_from_registry(HKEY_LOCAL_MACHINE, "SOFTWARE\\PARSEC", NULL);
    read_keys_from_registry(HKEY_CURRENT_USER, "SOFTWARE\\PARSEC", NULL);
#endif  /* defined(__WINDOWS__) */
    free(files);
    free(new_files);

    return PARSEC_SUCCESS;
}


/*
 * Register an integer MCA parameter that is not associated with a
 * component
 */
int parsec_mca_param_reg_int_name(const char *type,
                                 const char *param_name,
                                 const char *help_msg,
                                 bool internal,
                                 bool read_only,
                                 int default_value,
                                 int *current_value)
{
    int ret;
    parsec_mca_param_storage_t storage;
    parsec_mca_param_storage_t lookup;

    storage.intval = default_value;
    ret = param_register(type, NULL, param_name, help_msg,
                         PARSEC_MCA_PARAM_TYPE_INT, internal, read_only,
                         &storage, NULL, NULL, &lookup);
    if (ret >= 0 && NULL != current_value) {
        *current_value = lookup.intval;
    }
    return ret;
}

/*
 * Register a size_t MCA parameter that is not associated with a
 * component
 */
int parsec_mca_param_reg_sizet_name(const char *type,
                                   const char *param_name,
                                   const char *help_msg,
                                   bool internal,
                                   bool read_only,
                                   size_t default_value,
                                   size_t *current_value)
{
    int ret;
    parsec_mca_param_storage_t storage;
    parsec_mca_param_storage_t lookup;

    storage.sizetval = default_value;
    ret = param_register(type, NULL, param_name, help_msg,
                         PARSEC_MCA_PARAM_TYPE_SIZET, internal, read_only,
                         &storage, NULL, NULL, &lookup);
    if (ret >= 0 && NULL != current_value) {
        *current_value = lookup.sizetval;
    }
    return ret;
}

/*
 * Register a string MCA parameter that is not associated with a
 * component
 */
int parsec_mca_param_reg_string_name(const char *type,
                                    const char *param_name,
                                    const char *help_msg,
                                    bool internal,
                                    bool read_only,
                                    const char *default_value,
                                    char **current_value)
{
    int ret;
    parsec_mca_param_storage_t storage;
    parsec_mca_param_storage_t lookup;

    if (NULL != default_value) {
        storage.stringval = (char *) default_value;
    } else {
        storage.stringval = NULL;
    }
    ret = param_register(type, NULL, param_name, help_msg,
                         PARSEC_MCA_PARAM_TYPE_STRING, internal, read_only,
                         &storage, NULL, NULL,
                         (NULL != current_value) ? &lookup : NULL);
    if (ret >= 0 && NULL != current_value) {
        *current_value = lookup.stringval;
    }
    return ret;
}

/*
 * Register a synonym name for an existing MCA parameter
 */
int parsec_mca_param_reg_syn_name(int index_orig,
                                 const char *syn_type_name,
                                 const char *syn_param_name, bool deprecated)
{
    return syn_register(index_orig, syn_type_name, NULL,
                        syn_param_name, deprecated);
}

/*
 * Look up an integer MCA parameter.
 */
int parsec_mca_param_lookup_int(int index, int *value)
{
  parsec_mca_param_storage_t storage;

  if (param_lookup(index, &storage, NULL, NULL)) {
    *value = storage.intval;
    return PARSEC_SUCCESS;
  }
  return PARSEC_ERROR;
}

/*
 * Look up a size_t MCA parameter.
 */
int parsec_mca_param_lookup_sizet(int index, size_t *value)
{
  parsec_mca_param_storage_t storage;

  if (param_lookup(index, &storage, NULL, NULL)) {
    *value = storage.sizetval;
    return PARSEC_SUCCESS;
  }
  return PARSEC_ERROR;
}


/*
 * Set an integer parameter
 */
int parsec_mca_param_set_int(int index, int value)
{
    parsec_mca_param_storage_t storage;

    parsec_mca_param_unset(index);
    storage.intval = value;
    param_set_override(index, &storage, PARSEC_MCA_PARAM_TYPE_INT);
    return PARSEC_SUCCESS;
}

/*
 * Set a size_t parameter
 */
int parsec_mca_param_set_sizet(int index, size_t value)
{
    parsec_mca_param_storage_t storage;

    parsec_mca_param_unset(index);
    storage.sizetval = value;
    param_set_override(index, &storage, PARSEC_MCA_PARAM_TYPE_SIZET);
    return PARSEC_SUCCESS;
}

/*
 * Deregister a parameter
 */
int parsec_mca_param_deregister(int index)
{
    parsec_mca_param_t *array;
    size_t size;

    /* Lookup the index and see if the index and parameter are valid */
    size = parsec_value_array_get_size(&mca_params);
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    if (index < 0 || ((size_t) index) > size ||
        PARSEC_MCA_PARAM_TYPE_MAX >= array[index].mbp_type) {
        return PARSEC_ERROR;
    }

    /* Do not remove this item from the array otherwise we will change
       all the indices of parameters with a larger index. The destructor
       will mark this parameter as invalid. */
    OBJ_DESTRUCT(&array[index]);

    return PARSEC_SUCCESS;
}

/*
 * Look up a string MCA parameter.
 */
int parsec_mca_param_lookup_string(int index, char **value)
{
  parsec_mca_param_storage_t storage;

  if (param_lookup(index, &storage, NULL, NULL)) {
    *value = storage.stringval;
    return PARSEC_SUCCESS;
  }
  return PARSEC_ERROR;
}


/*
 * Set an string parameter
 */
int parsec_mca_param_set_string(int index, char *value)
{
    parsec_mca_param_storage_t storage;

    parsec_mca_param_unset(index);
    storage.stringval = value;
    param_set_override(index, &storage, PARSEC_MCA_PARAM_TYPE_STRING);
    return PARSEC_SUCCESS;
}


/*
 * Lookup the source of an MCA param's value
 */
int parsec_mca_param_lookup_source(int index, parsec_mca_param_source_t *source, char **source_file)
{
    parsec_mca_param_storage_t storage;
    int rc;

    storage.stringval = NULL;

    rc = param_lookup(index, &storage, source, source_file);
    if (PARSEC_MCA_PARAM_TYPE_STRING == param_type_from_index (index) &&
        NULL != storage.stringval) {
        free (storage.stringval);
    }

    return rc ? PARSEC_SUCCESS : PARSEC_ERROR;
}

/*
 * Unset a parameter
 */
int parsec_mca_param_unset(int index)
{
    size_t len;
    parsec_mca_param_t *array;

    if (!initialized) {
        return PARSEC_ERROR;
    }

    len = parsec_value_array_get_size(&mca_params);
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    if (index < 0 || ((size_t) index) > len ||
        PARSEC_MCA_PARAM_TYPE_MAX <= array[index].mbp_type) {
        return PARSEC_ERROR;
    }

    /* We have a valid entry so save the internal flag */
    if (array[index].mbp_override_value_set) {
        if (PARSEC_MCA_PARAM_TYPE_STRING == array[index].mbp_type &&
            NULL != array[index].mbp_override_value.stringval) {
            free(array[index].mbp_override_value.stringval);
            array[index].mbp_override_value.stringval = NULL;
        }
    }
    array[index].mbp_override_value_set = false;

    /* All done */

    return PARSEC_SUCCESS;
}


char *parsec_mca_param_env_var(const char *param_name)
{
    int rc;
    char *name;

    rc = asprintf(&name, "%s%s", mca_prefix, param_name);
    if (-1 == rc) {
        return NULL;
    }

    return name;
}

/*
 * Find the index for an MCA parameter based on its names.
 */
int parsec_mca_param_find(const char *type_name, const char *component_name,
                         const char *param_name)
{
  size_t i, size;
  parsec_mca_param_t *array;

  /* Check for bozo cases */

  if (!initialized) {
    return PARSEC_ERROR;
  }

  /* Loop through looking for a parameter of a given
     type/component/param */

  size = parsec_value_array_get_size(&mca_params);
  array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
  for (i = 0; i < size; ++i) {
    if (((NULL == type_name && NULL == array[i].mbp_type_name) ||
         (NULL != type_name && NULL != array[i].mbp_type_name &&
          (0 == strcmp(type_name, array[i].mbp_type_name)))) &&
        ((NULL == component_name && NULL == array[i].mbp_component_name) ||
         (NULL != component_name && NULL != array[i].mbp_component_name &&
          0 == strcmp(component_name, array[i].mbp_component_name))) &&
        ((NULL == param_name && NULL == array[i].mbp_param_name) ||
         (NULL != param_name && NULL != array[i].mbp_param_name &&
          0 == strcmp(param_name, array[i].mbp_param_name)))) {
      return (int)i;
    }
  }

  /* Didn't find it */

  return PARSEC_ERROR;
}


int parsec_mca_param_set_internal(int index, bool internal)
{
    size_t len;
    parsec_mca_param_t *array;

    /* Check for bozo cases */

    if (!initialized) {
        return PARSEC_ERROR;
    }

    len = parsec_value_array_get_size(&mca_params);
    if (((size_t) index) > len) {
        return PARSEC_ERROR;
    }

    /* We have a valid entry (remember that we never delete MCA
       parameters, so if the index is >0 and <len, it must be good),
       so save the internal flag */

    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    array[index].mbp_internal = internal;

    /* All done */

    return PARSEC_SUCCESS;
}


/*
 * Return a list of info of all currently registered parameters
 */
int parsec_mca_param_dump(parsec_list_t **info, bool internal)
{
    size_t i, j, len;
    parsec_mca_param_info_t *p, *q;
    parsec_mca_param_t *array;
    parsec_list_item_t *item;
    parsec_syn_info_t *si;

    /* Check for bozo cases */

    if (!initialized) {
        return PARSEC_ERROR;
    }

    if (NULL == info) {
        return PARSEC_ERROR;
    }
    *info = OBJ_NEW(parsec_list_t);

    /* Iterate through all the registered parameters */

    len = parsec_value_array_get_size(&mca_params);
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    for (i = 0; i < len; ++i) {
        if ((array[i].mbp_internal == internal || internal) &&
            PARSEC_MCA_PARAM_TYPE_MAX > array[i].mbp_type) {
            p = OBJ_NEW(parsec_mca_param_info_t);
            if (NULL == p) {
                return PARSEC_ERR_OUT_OF_RESOURCE;
            }
            p->mbpp_index = (int)i;
            p->mbpp_type_name = array[i].mbp_type_name;
            p->mbpp_component_name = array[i].mbp_component_name;
            p->mbpp_param_name = array[i].mbp_param_name;
            p->mbpp_full_name = array[i].mbp_full_name;
            p->mbpp_deprecated = array[i].mbp_deprecated;
            p->mbpp_internal = array[i].mbp_internal;
            p->mbpp_read_only = array[i].mbp_read_only;
            p->mbpp_type = array[i].mbp_type;
            p->mbpp_help_msg = array[i].mbp_help_msg;

            /* Save this entry to the list */
            parsec_list_append(*info, (parsec_list_item_t*) p);

            /* If this param has synonyms, add them too */
            if (NULL != array[i].mbp_synonyms &&
                !parsec_list_is_empty(array[i].mbp_synonyms)) {
                for (p->mbpp_synonyms_len = 0, item = PARSEC_LIST_ITERATOR_FIRST(array[i].mbp_synonyms);
                     PARSEC_LIST_ITERATOR_END(array[i].mbp_synonyms) != item;
                     ++p->mbpp_synonyms_len, item = PARSEC_LIST_ITERATOR_NEXT(item));
                p->mbpp_synonyms = malloc(sizeof(parsec_mca_param_info_t*) *
                                          p->mbpp_synonyms_len);
                if (NULL == p->mbpp_synonyms) {
                    p->mbpp_synonyms_len = 0;
                    return PARSEC_ERR_OUT_OF_RESOURCE;
                }

                for (j = 0, item = PARSEC_LIST_ITERATOR_FIRST(array[i].mbp_synonyms);
                     PARSEC_LIST_ITERATOR_END(array[i].mbp_synonyms) != item;
                     ++j, item = PARSEC_LIST_ITERATOR_NEXT(item)) {
                    si = (parsec_syn_info_t*) item;
                    q = OBJ_NEW(parsec_mca_param_info_t);
                    if (NULL == q) {
                        return PARSEC_ERR_OUT_OF_RESOURCE;
                    }
                    q->mbpp_index = (int)i;
                    q->mbpp_type_name = si->si_type_name;
                    q->mbpp_component_name = si->si_component_name;
                    q->mbpp_param_name = si->si_param_name;
                    q->mbpp_full_name = si->si_full_name;
                    q->mbpp_deprecated = si->si_deprecated ||
                        array[i].mbp_deprecated;
                    q->mbpp_internal = array[i].mbp_internal;
                    q->mbpp_read_only = array[i].mbp_read_only;
                    q->mbpp_type = array[i].mbp_type;
                    q->mbpp_help_msg = array[i].mbp_help_msg;

                    /* Let this one point to the original */
                    q->mbpp_synonym_parent = p;

                    /* Let the original point to this one */
                    p->mbpp_synonyms[j] = q;

                    /* Save this entry to the list */
                    parsec_list_append(*info, (parsec_list_item_t*) q);
                }
            }
        }
    }

    /* All done */

    return PARSEC_SUCCESS;
}


/*
 * Make an argv-style list of strings suitable for an environment
 */
int parsec_mca_param_build_env(char ***env, int *num_env, bool internal)
{
    int rc;
    size_t i, len;
    parsec_mca_param_t *array;
    char *str;
    parsec_mca_param_storage_t storage;

    /* Check for bozo cases */

    if (!initialized) {
        return PARSEC_ERROR;
    }

    /* Iterate through all the registered parameters */

    len = parsec_value_array_get_size(&mca_params);
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    for (i = 0; i < len; ++i) {
        /* Don't output read-only values */
        if (array[i].mbp_read_only) {
            continue;
        }

        if (array[i].mbp_internal == internal || internal) {
            if (param_lookup(i, &storage, NULL, NULL)) {
                if (PARSEC_MCA_PARAM_TYPE_INT == array[i].mbp_type) {
                    rc = asprintf(&str, "%s=%d", array[i].mbp_env_var_name,
                                  storage.intval);
                    if (-1 == rc) {
                        return PARSEC_ERR_OUT_OF_RESOURCE;
                    }
                    parsec_argv_append(num_env, env, str);
                    free(str);
                } else if (PARSEC_MCA_PARAM_TYPE_SIZET == array[i].mbp_type) {
                    rc = asprintf(&str, "%s=%lu", array[i].mbp_env_var_name,
                                  (unsigned long)storage.sizetval);
                    if (-1 == rc) {
                        return PARSEC_ERR_OUT_OF_RESOURCE;
                    }
                    parsec_argv_append(num_env, env, str);
                    free(str);
                } else if (PARSEC_MCA_PARAM_TYPE_STRING == array[i].mbp_type) {
                    if (NULL != storage.stringval) {
                        rc = asprintf(&str, "%s=%s", array[i].mbp_env_var_name,
                                      storage.stringval);
                        if (-1 == rc) {
                            return PARSEC_ERR_OUT_OF_RESOURCE;
                        }
                        free(storage.stringval);
                        parsec_argv_append(num_env, env, str);
                        free(str);
                    }
                } else {
                    goto cleanup;
                }
            } else {
                goto cleanup;
            }
        }
    }

    /* All done */

    return PARSEC_SUCCESS;

    /* Error condition */

 cleanup:
    if (*num_env > 0) {
        parsec_argv_free(*env);
        *num_env = 0;
        *env = NULL;
    }
    (void)rc;
    return PARSEC_ERR_NOT_FOUND;
}


/*
 * Free a list -- and all associated memory -- that was previously
 * returned from mca_param_dump()
 */
int parsec_mca_param_dump_release(parsec_list_t *info)
{
    parsec_list_item_t *item;

    for (item = parsec_list_pop_front(info); NULL != item;
         item = parsec_list_pop_front(info)) {
        OBJ_RELEASE(item);
    }
    OBJ_RELEASE(info);

    return PARSEC_SUCCESS;
}


/*
 * Shut down the MCA parameter system (normally only invoked by the
 * MCA framework itself).
 */
int parsec_mca_param_finalize(void)
{
    parsec_list_item_t *item;
    parsec_mca_param_t *array;

    if (initialized) {
        int size, i;

        size = parsec_value_array_get_size(&mca_params);
        array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
        for (i = 0 ; i < size ; ++i) {
            if (PARSEC_MCA_PARAM_TYPE_MAX > array[i].mbp_type) {
                OBJ_DESTRUCT(&array[i]);
            }
        }
        OBJ_DESTRUCT(&mca_params);

        while (NULL !=
               (item = parsec_list_pop_front(&parsec_mca_param_file_values))) {
            OBJ_RELEASE(item);
        }
        OBJ_DESTRUCT(&parsec_mca_param_file_values);

        initialized = false;
    }

    /* All done */

    return PARSEC_SUCCESS;
}

static int read_files(char *file_list)
{
    int i, count;
    char **files;

    /* Iterate through all the files passed in -- read them in reverse
       order so that we preserve unix/shell path-like semantics (i.e.,
       the entries farthest to the left get precedence) */

    files = parsec_argv_split(file_list, PARSEC_ENV_SEP);
    count = parsec_argv_count(files);

    parsec_util_keyval_parse_init();
    for (i = count - 1; i >= 0; --i) {
        parsec_mca_parse_paramfile(files[i]);
    }
    parsec_util_keyval_parse_finalize();
    parsec_argv_free(files);

    return PARSEC_SUCCESS;
}

/**
 *
 */
#if defined(__WINDOWS__)
#define MAX_KEY_LENGTH 255
#define MAX_VALUE_NAME 16383

static int read_keys_from_registry(HKEY hKey, char *sub_key, char *current_name)
{
    TCHAR   achKey[MAX_KEY_LENGTH];        /* buffer for subkey name */
    DWORD   cbName;                        /* size of name string */
    TCHAR   achClass[MAX_PATH] = TEXT(""); /* buffer for class name */
    DWORD   cchClassName = MAX_PATH;       /* size of class string */
    DWORD   cSubKeys=0;                    /* number of subkeys */
    DWORD   cbMaxSubKey;                   /* longest subkey size */
    DWORD   cchMaxClass;                   /* longest class string */
    DWORD   cValues;                       /* number of values for key */
    DWORD   cchMaxValue;                   /* longest value name */
    DWORD   cbMaxValueData;                /* longest value data */
    DWORD   cbSecurityDescriptor;          /* size of security descriptor */

    LPDWORD lpType;
    LPDWORD word_lpData;
    TCHAR   str_lpData[MAX_VALUE_NAME];
    TCHAR   *str_key_name, *type_name, *next_name;
    DWORD   dwSize, i, retCode, type_len, param_type;
    TCHAR achValue[MAX_VALUE_NAME];
    DWORD cchValue = MAX_VALUE_NAME;
    HKEY hTestKey;
    char *sub_sub_key;
    parsec_mca_param_storage_t storage, override, lookup;
    int rc;

    if( !RegOpenKeyEx( hKey, sub_key, 0, KEY_READ, &hTestKey) == ERROR_SUCCESS )
        return PARSEC_ERROR;

    /* Get the class name and the value count. */
    retCode = RegQueryInfoKey( hTestKey,                /* key handle */
                               achClass,                /* buffer for class name */
                               &cchClassName,           /* size of class string */
                               NULL,                    /* reserved */
                               &cSubKeys,               /* number of subkeys */
                               &cbMaxSubKey,            /* longest subkey size */
                               &cchMaxClass,            /* longest class string */
                               &cValues,                /* number of values for this key */
                               &cchMaxValue,            /* longest value name */
                               &cbMaxValueData,         /* longest value data */
                               &cbSecurityDescriptor,   /* security descriptor */
                               NULL );

    /* Enumerate the subkeys, until RegEnumKeyEx fails. */
    for (i = 0; i < cSubKeys; i++) {
        cbName = MAX_KEY_LENGTH;
        retCode = RegEnumKeyEx(hTestKey, i, achKey, &cbName, NULL, NULL, NULL, NULL);
        if (retCode != ERROR_SUCCESS) continue;
        asprintf(&sub_sub_key, "%s\\%s", sub_key, achKey);
        assert(0 < rc);
        if( NULL != current_name ) {
            asprintf(&next_name, "%s_%s", current_name, achKey);
            assert(0 < rc);
        } else {
            asprintf(&next_name, "%s", achKey);
            assert(0 < rc);
        }
        read_keys_from_registry(hKey, sub_sub_key, next_name);
        free(next_name);
        free(sub_sub_key);
    }

    /* Enumerate the key values. */
    for( i = 0; i < cValues; i++ ) {
        cchValue = MAX_VALUE_NAME;
        achValue[0] = '\0';
        retCode = RegEnumValue(hTestKey, i, achValue, &cchValue, NULL, NULL, NULL, NULL);
        if (retCode != ERROR_SUCCESS ) continue;

        /* lpType - get the type of the value
         * dwSize - get the size of the buffer to hold the value
         */
        retCode = RegQueryValueEx(hTestKey, achValue, NULL, (LPDWORD)&lpType, NULL, &dwSize);

        if (strcmp(achValue,"")) {
            if (current_name!=NULL) {
                rc = asprintf(&type_name, "%s_%s", current_name, achValue);
                assert(0 < rc);
            } else {
                rc = asprintf(&type_name, "%s", achValue);
                assert(0 < rc);
            }
        } else {
            if (current_name!=NULL) {
                rc = asprintf(&type_name, "%s", current_name);
                assert(0 < rc);
            } else {
                rc = asprintf(&type_name, "%s", achValue);
                assert(0 < rc);
            }
        }
        if (-1 == rc) {
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }

        type_len = strcspn(type_name, "_");
        str_key_name = type_name + type_len + 1;
        if( type_len == strlen(type_name) )
            str_key_name = NULL;
        type_name[type_len] = '\0';

        retCode = 1;
        if( lpType == (LPDWORD)REG_SZ ) { /* REG_SZ = 1 */
            retCode = RegQueryValueEx(hTestKey, achValue, NULL, NULL, (LPBYTE)&str_lpData, &dwSize);
            storage.stringval = (char*)str_lpData;
            override.stringval = (char*)str_lpData;
            param_type = PARSEC_MCA_PARAM_TYPE_STRING;
        } else if( lpType == (LPDWORD)REG_DWORD ) { /* REG_DWORD = 4 */
            retCode = RegQueryValueEx(hTestKey, achValue, NULL, NULL, (LPBYTE)&word_lpData, &dwSize);
            storage.intval  = (int)word_lpData;
            override.intval = (int)word_lpData;
            param_type = PARSEC_MCA_PARAM_TYPE_INT;
            /* PARSEC_MCA_PARAM_TYPE_SIZET not supported */
        }
        if( !retCode ) {
            (void)param_register( type_name, NULL, str_key_name, NULL,
                                  param_type, false, false,
                                  &storage, NULL, &override, &lookup );
        } else {
            parsec_output( 0, "error reading value of param_name: %s with %d error.\n",
                         str_key_name, retCode);
        }

        free(type_name);
    }

    RegCloseKey( hKey );

    return PARSEC_SUCCESS;
}
#endif  /* defined(__WINDOWS__) */

/******************************************************************************/

static int param_register(const char *type_name,
                          const char *component_name,
                          const char *param_name,
                          const char *help_msg,
                          parsec_mca_param_type_t type,
                          bool internal,
                          bool read_only,
                          parsec_mca_param_storage_t *default_value,
                          parsec_mca_param_storage_t *file_value,
                          parsec_mca_param_storage_t *override_value,
                          parsec_mca_param_storage_t *current_value)
{
    int ret;
    size_t i, len;
    parsec_mca_param_t param, *array;

    /* Initialize the array if it has never been initialized */

    if (!initialized) {
        parsec_mca_param_init();
    }

    /* Create a parameter entry */

    OBJ_CONSTRUCT(&param, parsec_mca_param_t);
    param.mbp_type = type;
    param.mbp_internal = internal;
    param.mbp_read_only = read_only;
    if (NULL != help_msg) {
        param.mbp_help_msg = strdup(help_msg);
    }

    if (NULL != type_name) {
        param.mbp_type_name = strdup(type_name);
        if (NULL == param.mbp_type_name) {
            OBJ_DESTRUCT(&param);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }
    if (NULL != component_name) {
        param.mbp_component_name = strdup(component_name);
        if (NULL == param.mbp_component_name) {
            OBJ_DESTRUCT(&param);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }
    param.mbp_param_name = NULL;
    if (NULL != param_name) {
        param.mbp_param_name = strdup(param_name);
        if (NULL == param.mbp_param_name) {
            OBJ_DESTRUCT(&param);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    /* Build up the full name */
    len = 16;
    if (NULL != type_name) {
        len += strlen(type_name);
    }
    if (NULL != param.mbp_component_name) {
        len += strlen(param.mbp_component_name);
    }
    if (NULL != param.mbp_param_name) {
        len += strlen(param.mbp_param_name);
    }

    param.mbp_full_name = (char*)malloc(len);
    if (NULL == param.mbp_full_name) {
        OBJ_DESTRUCT(&param);
        return PARSEC_ERROR;
    }

    /* Copy the name over in parts */

    param.mbp_full_name[0] = '\0';
    if (NULL != type_name) {
        strncat(param.mbp_full_name, type_name, len-1);
    }
    if (NULL != component_name) {
        if ('\0' != param.mbp_full_name[0]) {
            strcat(param.mbp_full_name, "_");
        }
        strcat(param.mbp_full_name, component_name);
    }
    if (NULL != param_name) {
        if ('\0' != param.mbp_full_name[0]) {
            strcat(param.mbp_full_name, "_");
        }
        strcat(param.mbp_full_name, param_name);
    }

    /* Create the environment name */

    len = strlen(param.mbp_full_name) + strlen(mca_prefix) + 16;
    param.mbp_env_var_name = (char*)malloc(len);
    if (NULL == param.mbp_env_var_name) {
        OBJ_DESTRUCT(&param);
        return PARSEC_ERROR;
    }
    snprintf(param.mbp_env_var_name, len, "%s%s", mca_prefix,
             param.mbp_full_name);

    /* Figure out the default value; zero it out if a default is not
     provided */

    if (NULL != default_value) {
        if (PARSEC_MCA_PARAM_TYPE_STRING == param.mbp_type &&
            NULL != default_value->stringval) {
            param.mbp_default_value.stringval = strdup(default_value->stringval);
        } else {
            param.mbp_default_value = *default_value;
        }
    } else {
        memset(&param.mbp_default_value, 0, sizeof(param.mbp_default_value));
    }

    /* Figure out the file value; zero it out if a file is not
     provided */

    if (NULL != file_value) {
        if (PARSEC_MCA_PARAM_TYPE_STRING == param.mbp_type &&
            NULL != file_value->stringval) {
            param.mbp_file_value.stringval = strdup(file_value->stringval);
        } else {
            param.mbp_file_value = *file_value;
        }
        param.mbp_file_value_set = true;
    } else {
        memset(&param.mbp_file_value, 0, sizeof(param.mbp_file_value));
        param.mbp_file_value_set = false;
    }

    /* Figure out the override value; zero it out if a override is not
     provided */

    if (NULL != override_value) {
        if (PARSEC_MCA_PARAM_TYPE_STRING == param.mbp_type &&
            NULL != override_value->stringval) {
            param.mbp_override_value.stringval = strdup(override_value->stringval);
        } else {
            param.mbp_override_value = *override_value;
        }
        param.mbp_override_value_set = true;
    } else {
        memset(&param.mbp_override_value, 0, sizeof(param.mbp_override_value));
        param.mbp_override_value_set = false;
    }

    /* See if this entry is already in the array */

    len = parsec_value_array_get_size(&mca_params);
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    for (i = 0; i < len; ++i) {
        if (0 == strcmp(param.mbp_full_name, array[i].mbp_full_name)) {

            /* We found an entry with the same param name.  Check to
             ensure that we're not changing types */
            /* Easy case: both are INT */

            if (PARSEC_MCA_PARAM_TYPE_INT == array[i].mbp_type &&
                PARSEC_MCA_PARAM_TYPE_INT == param.mbp_type) {
                if (NULL != default_value) {
                    array[i].mbp_default_value.intval =
                        param.mbp_default_value.intval;
                }
                if (NULL != file_value) {
                    array[i].mbp_file_value.intval =
                        param.mbp_file_value.intval;
                    array[i].mbp_file_value_set = true;
                }
                if (NULL != override_value) {
                    array[i].mbp_override_value.intval =
                        param.mbp_override_value.intval;
                    array[i].mbp_override_value_set = true;
                }
            }

            /* Both are SIZE_T */

            else if (PARSEC_MCA_PARAM_TYPE_SIZET == array[i].mbp_type &&
                     PARSEC_MCA_PARAM_TYPE_SIZET == param.mbp_type) {
                if (NULL != default_value) {
                    array[i].mbp_default_value.sizetval =
                        param.mbp_default_value.sizetval;
                }
                if (NULL != file_value) {
                    array[i].mbp_file_value.sizetval =
                        param.mbp_file_value.sizetval;
                    array[i].mbp_file_value_set = true;
                }
                if (NULL != override_value) {
                    array[i].mbp_override_value.sizetval =
                        param.mbp_override_value.sizetval;
                    array[i].mbp_override_value_set = true;
                }
            }

            /* Both are STRING */

            else if (PARSEC_MCA_PARAM_TYPE_STRING == array[i].mbp_type &&
                     PARSEC_MCA_PARAM_TYPE_STRING == param.mbp_type) {
                if (NULL != default_value) {
                    if (NULL != array[i].mbp_default_value.stringval) {
                        free(array[i].mbp_default_value.stringval);
                        array[i].mbp_default_value.stringval = NULL;
                    }
                    if (NULL != param.mbp_default_value.stringval) {
                        array[i].mbp_default_value.stringval =
                            strdup(param.mbp_default_value.stringval);
                    }
                }

                if (NULL != file_value) {
                    if (NULL != array[i].mbp_file_value.stringval) {
                        free(array[i].mbp_file_value.stringval);
                        array[i].mbp_file_value.stringval = NULL;
                    }
                    if (NULL != param.mbp_file_value.stringval) {
                        array[i].mbp_file_value.stringval =
                            strdup(param.mbp_file_value.stringval);
                    }
                    array[i].mbp_file_value_set = true;
                }

                if (NULL != override_value) {
                    if (NULL != array[i].mbp_override_value.stringval) {
                        free(array[i].mbp_override_value.stringval);
                        array[i].mbp_override_value.stringval = NULL;
                    }
                    if (NULL != param.mbp_override_value.stringval) {
                        array[i].mbp_override_value.stringval =
                            strdup(param.mbp_override_value.stringval);
                    }
                    array[i].mbp_override_value_set = true;
                }
            }

            /* If the original is INT and the new is STRING, or the original
             is STRING and the new is INT, this is an developer error. */

            else if (param.mbp_type != array[i].mbp_type) {
#if defined(PARSEC_DEBUG)
                parsec_show_help("help-mca-param.txt",
                                "re-register with different type",
                                true, array[i].mbp_full_name);
#endif
                /* Return an error code and hope for the best. */
                OBJ_DESTRUCT(&param);
                return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
            }

            /* Now delete the newly-created entry (since we just saved the
             value in the old entry) */

            OBJ_DESTRUCT(&param);

            /* Finally, if we have a lookup value, look it up */

            if (NULL != current_value) {
                if (!param_lookup(i, current_value, NULL, NULL)) {
                    return PARSEC_ERR_NOT_FOUND;
                }
            }

            /* Return the new index */

            return (int)i;
        }
    }

    /* Add it to the array.  Note that we copy the parsec_mca_param_t by value,
     so the entire contents of the struct is copied.  The synonym list
     will always be empty at this point, so there's no need for an
     extra RETAIN or RELEASE. */
    if (PARSEC_SUCCESS !=
        (ret = parsec_value_array_append_item(&mca_params, &param))) {
        return ret;
    }
    ret = (int)parsec_value_array_get_size(&mca_params) - 1;

    /* Finally, if we have a lookup value, look it up */

    if (NULL != current_value) {
        if (!param_lookup(ret, current_value, NULL, NULL)) {
            return PARSEC_ERR_NOT_FOUND;
        }
    }

    /* All done */

    return ret;
}


/*
 * Back-end for registering a synonym
 */
static int syn_register(int index_orig, const char *syn_type_name,
                        const char *syn_component_name,
                        const char *syn_param_name, bool deprecated)
{
    size_t len;
    parsec_syn_info_t *si;
    parsec_mca_param_t *array;

    if (!initialized) {
        return PARSEC_ERROR;
    }

    /* Sanity check index param */
    len = parsec_value_array_get_size(&mca_params);
    if (index_orig < 0 || ((size_t) index_orig) > len) {
        return PARSEC_ERR_BAD_PARAM;
    }

    /* Make the synonym info object */
    si = OBJ_NEW(parsec_syn_info_t);
    if (NULL == si) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    /* Note that the following logic likely could have been combined
     into more compact code.  However, keeping it separate made it
     much easier to read / maintain (IMHO).  This is not a high
     performance section of the code, so a premium was placed on
     future readability / maintenance. */

    /* Save the function parameters */
    si->si_deprecated = deprecated;
    if (NULL != syn_type_name) {
        si->si_type_name = strdup(syn_type_name);
        if (NULL == si->si_type_name) {
            OBJ_RELEASE(si);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    if (NULL != syn_component_name) {
        si->si_component_name = strdup(syn_component_name);
        if (NULL == si->si_component_name) {
            OBJ_RELEASE(si);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    if (NULL != syn_param_name) {
        si->si_param_name = strdup(syn_param_name);
        if (NULL == si->si_param_name) {
            OBJ_RELEASE(si);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    /* Build up the full name */
    len = 16;
    if (NULL != syn_type_name) {
        len += strlen(syn_type_name);
    }
    if (NULL != syn_component_name) {
        len += strlen(syn_component_name);
    }
    if (NULL != syn_param_name) {
        len += strlen(syn_param_name);
    }
    si->si_full_name = (char*) malloc(len);
    if (NULL == si->si_full_name) {
        OBJ_RELEASE(si);
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    /* Copy the name over in parts */
    si->si_full_name[0] = '\0';
    if (NULL != syn_type_name) {
        strncat(si->si_full_name, syn_type_name, len-1);
    }
    if (NULL != syn_component_name) {
        if ('\0' != si->si_full_name[0]) {
            strcat(si->si_full_name, "_");
        }
        strcat(si->si_full_name, syn_component_name);
    }
    if (NULL != syn_param_name) {
        if ('\0' != si->si_full_name[0]) {
            strcat(si->si_full_name, "_");
        }
        strcat(si->si_full_name, syn_param_name);
    }

    /* Create the environment name */
    len = strlen(si->si_full_name) + strlen(mca_prefix) + 16;
    si->si_env_var_name = (char*) malloc(len);
    if (NULL == si->si_env_var_name) {
        OBJ_RELEASE(si);
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    snprintf(si->si_env_var_name, len, "%s%s", mca_prefix,
             si->si_full_name);

    /* Find the param entry; add this syn_info to its list of
     synonyms */
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);

    /* Sanity check. Is this a valid parameter? */
    if (PARSEC_MCA_PARAM_TYPE_MAX <= array[index_orig].mbp_type) {
        OBJ_RELEASE(si);
        return PARSEC_ERROR;
    }

    if (NULL == array[index_orig].mbp_synonyms) {
        array[index_orig].mbp_synonyms = OBJ_NEW(parsec_list_t);
    }
    parsec_list_append(array[index_orig].mbp_synonyms, &(si->super));

    /* All done */

    return PARSEC_SUCCESS;
}


/*
 * Set an override
 */
static bool param_set_override(size_t index,
                               parsec_mca_param_storage_t *storage,
                               parsec_mca_param_type_t type)
{
    size_t size;
    parsec_mca_param_t *array;

    /* Lookup the index and see if it's valid */

    if (!initialized) {
        return false;
    }
    size = parsec_value_array_get_size(&mca_params);
    if (index > size) {
        return false;
    }

    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);
    if (PARSEC_MCA_PARAM_TYPE_INT == type) {
        array[index].mbp_override_value.intval = storage->intval;
    } else if (PARSEC_MCA_PARAM_TYPE_SIZET == type) {
        array[index].mbp_override_value.sizetval = storage->sizetval;
    } else if (PARSEC_MCA_PARAM_TYPE_STRING == type) {
        if (NULL != storage->stringval) {
            array[index].mbp_override_value.stringval =
                strdup(storage->stringval);
        } else {
            array[index].mbp_override_value.stringval = NULL;
        }
    } else {
        return false;
    }

    array[index].mbp_override_value_set = true;

    return true;
}

/*
 * Lookup the type of a parameter from an index
 */
static parsec_mca_param_type_t param_type_from_index (size_t index)
{
    parsec_mca_param_t *array;
    size_t size;

    /* Lookup the index and see if it's valid */

    if (!initialized) {
        return PARSEC_MCA_PARAM_ERROR;
    }
    size = parsec_value_array_get_size(&mca_params);
    if (index > size) {
        return PARSEC_MCA_PARAM_ERROR;
    }
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);

    return array[index].mbp_type;
}

/*
 * Lookup a parameter in multiple places
 */
static bool param_lookup(size_t index, parsec_mca_param_storage_t *storage,
                         parsec_mca_param_source_t *source_param,
                         char **source_file)
{
    int rc;
    size_t size;
    parsec_mca_param_t *array;
    char *p, *q;
    parsec_mca_param_source_t source = MCA_PARAM_SOURCE_MAX;

    /* default the value */
    if (NULL != source_file) {
        *source_file = NULL;
    }

    /* Lookup the index and see if it's valid */

    if (!initialized) {
        return false;
    }
    size = parsec_value_array_get_size(&mca_params);
    if (index > size) {
        return false;
    }
    array = PARSEC_VALUE_ARRAY_GET_BASE(&mca_params, parsec_mca_param_t);

    /* Ensure that MCA param has a good type */

    if (PARSEC_MCA_PARAM_TYPE_INT != array[index].mbp_type &&
        PARSEC_MCA_PARAM_TYPE_SIZET != array[index].mbp_type &&
        PARSEC_MCA_PARAM_TYPE_STRING != array[index].mbp_type) {
        return false;
    }

    /* Check all the places that the param may be hiding, in priority
       order -- but if read_only is true, then only look at the
       default location. */

    if (array[index].mbp_read_only) {
        if (lookup_override(&array[index], storage) ||
             lookup_env(&array[index], storage) ||
             lookup_file(&array[index], storage, source_file)) {
            parsec_show_help("help-mca-param.txt", "read-only-param-set",
                           true, array[index].mbp_full_name);
        }

        /* First look at the "real" name of this param */
        if (lookup_default(&array[index], storage)) {
            source = MCA_PARAM_SOURCE_DEFAULT;
        }
    } else {
        if (lookup_override(&array[index], storage)) {
            source = MCA_PARAM_SOURCE_OVERRIDE;
        } else if (lookup_env(&array[index], storage)) {
            source = MCA_PARAM_SOURCE_ENV;
        } else if (lookup_file(&array[index], storage, source_file)) {
            source = MCA_PARAM_SOURCE_FILE;
        } else if (lookup_default(&array[index], storage)) {
            source = MCA_PARAM_SOURCE_DEFAULT;
        }
    }
    if (MCA_PARAM_SOURCE_MAX != source) {
        if (NULL != source_param) {
            *source_param = source;
        }

        /* If we're returning a string, replace all instances of "~/"
           with the user's home directory */

        if (PARSEC_MCA_PARAM_TYPE_STRING == array[index].mbp_type &&
            NULL != storage->stringval) {
            if (0 == strncmp(storage->stringval, "~/", 2)) {
                if( NULL == home ) {
                    rc = asprintf(&p, "%s", storage->stringval + 2);
                    assert(0 < rc);
                } else {
                    p = parsec_os_path( false, home, storage->stringval + 2, NULL );
                }
                free(storage->stringval);
                storage->stringval = p;
            }

            p = strstr(storage->stringval, ":~/");
            while (NULL != p) {
                *p = '\0';
                if( NULL == home ) {
                    rc = asprintf(&q, "%s:%s", storage->stringval, p + 2);
                    assert(0 < rc);
                } else {
                    rc = asprintf(&q, "%s:%s%s", storage->stringval, home, p + 2);
                    assert(0 < rc);
                }
                free(storage->stringval);
                storage->stringval = q;
                p = strstr(storage->stringval, ":~/");
            }
        }

        return true;
    }

    /* Didn't find it.  Doh! */
    (void)rc;
    return false;
}


/*
 * Lookup a param in the overrides section
 */
static bool lookup_override(parsec_mca_param_t *param,
                            parsec_mca_param_storage_t *storage)
{
    if (param->mbp_override_value_set) {
        if (PARSEC_MCA_PARAM_TYPE_INT == param->mbp_type) {
            storage->intval = param->mbp_override_value.intval;
        } else if (PARSEC_MCA_PARAM_TYPE_SIZET == param->mbp_type) {
            storage->sizetval = param->mbp_override_value.sizetval;
        } else if (PARSEC_MCA_PARAM_TYPE_STRING == param->mbp_type) {
            storage->stringval = strdup(param->mbp_override_value.stringval);
        }

        return true;
    }

    /* Don't have an override */

    return false;
}


/*
 * Lookup a param in the environment
 */
static bool lookup_env(parsec_mca_param_t *param,
                       parsec_mca_param_storage_t *storage)
{
    char *env = NULL;
    parsec_list_item_t *item;
    parsec_syn_info_t *si;
    char *deprecated_name = NULL;
    bool print_deprecated_warning = false;

    /* Look for the primary param name */
    if (NULL != param->mbp_env_var_name) {
        env = getenv(param->mbp_env_var_name);
        print_deprecated_warning =
            param->mbp_deprecated & !param->mbp_deprecated_warning_shown;
        deprecated_name = param->mbp_full_name;
        /* Regardless of whether we want to show the deprecated
           warning or not, we can skip this check the next time
           through on this parameter */
        param->mbp_deprecated_warning_shown = true;
    }

    /* If we didn't find the primary name, look in all the synonyms */
    if (NULL == env && NULL != param->mbp_synonyms &&
        !parsec_list_is_empty(param->mbp_synonyms)) {
        for (item = PARSEC_LIST_ITERATOR_FIRST(param->mbp_synonyms);
             NULL == env && PARSEC_LIST_ITERATOR_END(param->mbp_synonyms) != item;
             item = PARSEC_LIST_ITERATOR_NEXT(item)) {
            si = (parsec_syn_info_t*) item;
            env = getenv(si->si_env_var_name);
            if (NULL != env &&
                ((si->si_deprecated &&
                  !si->si_deprecated_warning_shown) ||
                 (param->mbp_deprecated &&
                  !param->mbp_deprecated_warning_shown))) {
                print_deprecated_warning =
                    si->si_deprecated_warning_shown =
                    param->mbp_deprecated_warning_shown = true;
                deprecated_name = si->si_full_name;
            }
        }
    }

    /* If we found it, react */
    if (NULL != env) {
        if (PARSEC_MCA_PARAM_TYPE_INT == param->mbp_type) {
            storage->intval = (int)strtol(env,(char**)NULL,0);
        } else if (PARSEC_MCA_PARAM_TYPE_SIZET == param->mbp_type) {
            storage->sizetval = (size_t)strtoll(env,(char**)NULL,0);
        } else if (PARSEC_MCA_PARAM_TYPE_STRING == param->mbp_type) {
            storage->stringval = strdup(env);
        }

        if (print_deprecated_warning) {
            parsec_show_help("help-mca-param.txt", "deprecated mca param env",
                           true, deprecated_name);
        }
        return true;
    }

    /* Didn't find it */
    return false;
}


/*
 * Lookup a param in the files
 */
static bool lookup_file(parsec_mca_param_t *param,
                        parsec_mca_param_storage_t *storage,
                        char **source_file)
{
    bool found = false;
    parsec_syn_info_t *si;
    char *deprecated_name = NULL;
    parsec_list_item_t *item, *in_item;
    parsec_mca_param_file_value_t *fv;
    bool print_deprecated_warning = false;

    /* See if we previously found a match from a file.  If so, just
       return that */

    if (param->mbp_file_value_set) {
        if (NULL != source_file) {
            *source_file = param->mbp_source_file;
        }
        return set(param->mbp_type, storage, &param->mbp_file_value);
    }

    /* Scan through the list of values read in from files and try to
       find a match.  If we do, cache it on the param (for future
       lookups) and save it in the storage. */

    for (item = PARSEC_LIST_ITERATOR_FIRST(&parsec_mca_param_file_values);
         PARSEC_LIST_ITERATOR_END(&parsec_mca_param_file_values) != item;
         item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        fv = (parsec_mca_param_file_value_t *) item;
        /* If it doesn't match the parameter's real name, check its
           synonyms */
        if (0 == strcmp(fv->mbpfv_param, param->mbp_full_name)) {
            found = true;
            print_deprecated_warning =
                param->mbp_deprecated & !param->mbp_deprecated_warning_shown;
            deprecated_name = param->mbp_full_name;
            /* Regardless of whether we want to show the deprecated
               warning or not, we can skip this check the next time
               through on this parameter */
            param->mbp_deprecated_warning_shown = true;
        } else if (NULL != param->mbp_synonyms &&
                   !parsec_list_is_empty(param->mbp_synonyms)) {
            /* Check all the synonyms on this parameter and see if the
               file value matches */
            for (in_item = PARSEC_LIST_ITERATOR_FIRST(param->mbp_synonyms);
                 PARSEC_LIST_ITERATOR_END(param->mbp_synonyms) != in_item;
                 in_item = PARSEC_LIST_ITERATOR_NEXT(in_item)) {
                si = (parsec_syn_info_t*) in_item;
                if (0 == strcmp(fv->mbpfv_param, si->si_full_name)) {
                    found = true;
                    if ((si->si_deprecated &&
                         !si->si_deprecated_warning_shown) ||
                        (param->mbp_deprecated &&
                         !param->mbp_deprecated_warning_shown)) {
                        print_deprecated_warning =
                            si->si_deprecated_warning_shown =
                            param->mbp_deprecated_warning_shown = true;
                        deprecated_name = si->si_full_name;
                    }
                }
            }
        }

        /* Did we find it? */
        if (found) {
            if (PARSEC_MCA_PARAM_TYPE_INT == param->mbp_type) {
                if (NULL != fv->mbpfv_value) {
                    param->mbp_file_value.intval =
                        (int)strtol(fv->mbpfv_value,(char**)NULL,0);
                } else {
                    param->mbp_file_value.intval = 0;
                }
            } else if (PARSEC_MCA_PARAM_TYPE_SIZET == param->mbp_type) {
                if (NULL != fv->mbpfv_value) {
                    param->mbp_file_value.sizetval =
                        (size_t)strtoll(fv->mbpfv_value,(char**)NULL,0);
                } else {
                    param->mbp_file_value.sizetval = 0;
                }
            } else {
                param->mbp_file_value.stringval = fv->mbpfv_value;
                fv->mbpfv_value = NULL;
            }
            if (NULL != fv->mbpfv_file) {
                param->mbp_source_file = strdup(fv->mbpfv_file);
            }
            param->mbp_file_value_set = true;

            /* If the caller requested to know what file we found the
               value in, give them a copy of the filename pointer */
            if (NULL != source_file) {
                *source_file = param->mbp_source_file;
            }

            /* Since this is now cached on the param, we might as well
               remove it from the list and make future file lookups
               faster */

            parsec_list_nolock_remove(&parsec_mca_param_file_values,
                                     (parsec_list_item_t *)fv);
            OBJ_RELEASE(fv);

            /* Print the deprecated warning, if applicable */
            if (print_deprecated_warning) {
                parsec_show_help("help-mca-param.txt",
                               "deprecated mca param file",
                               true, deprecated_name);
            }

            return set(param->mbp_type, storage, &param->mbp_file_value);
        }
    }

    return false;
}


/*
 * Return the default value for a param
 */
static bool lookup_default(parsec_mca_param_t *param,
                           parsec_mca_param_storage_t *storage)
{
    return set(param->mbp_type, storage, &param->mbp_default_value);
}


static bool set(parsec_mca_param_type_t type,
                parsec_mca_param_storage_t *dest, parsec_mca_param_storage_t *src)
{
    switch (type) {
    case PARSEC_MCA_PARAM_TYPE_INT:
        dest->intval = src->intval;
        break;

    case PARSEC_MCA_PARAM_TYPE_SIZET:
        dest->sizetval = src->sizetval;
        break;

    case PARSEC_MCA_PARAM_TYPE_STRING:
        if (NULL != src->stringval) {
            dest->stringval = strdup(src->stringval);
        } else {
            dest->stringval = NULL;
        }
        break;

    default:
        return false;
        break;
    }

    return true;
}


/*
 * Create an empty param container
 */
static void param_constructor(parsec_mca_param_t *p)
{
    p->mbp_type = PARSEC_MCA_PARAM_TYPE_MAX;
    p->mbp_internal = false;
    p->mbp_read_only = false;
    p->mbp_deprecated = false;
    p->mbp_deprecated_warning_shown = false;

    p->mbp_type_name = NULL;
    p->mbp_component_name = NULL;
    p->mbp_param_name = NULL;
    p->mbp_full_name = NULL;
    p->mbp_help_msg = NULL;

    p->mbp_env_var_name = NULL;

    p->mbp_default_value.stringval = NULL;
    p->mbp_file_value_set = false;
    p->mbp_file_value.stringval = NULL;
    p->mbp_source_file = NULL;
    p->mbp_override_value_set = false;
    p->mbp_override_value.stringval = NULL;

    p->mbp_synonyms = NULL;
}


/*
 * Free all the contents of a param container
 */
static void param_destructor(parsec_mca_param_t *p)
{
    parsec_list_item_t *item;

    if (NULL != p->mbp_type_name) {
        free(p->mbp_type_name);
    }
    if (NULL != p->mbp_component_name) {
        free(p->mbp_component_name);
    }
    if (NULL != p->mbp_param_name) {
        free(p->mbp_param_name);
    }
    if (NULL != p->mbp_env_var_name) {
        free(p->mbp_env_var_name);
    }
    if (NULL != p->mbp_full_name) {
        free(p->mbp_full_name);
    }
    if (NULL != p->mbp_help_msg) {
        free(p->mbp_help_msg);
    }
    if (PARSEC_MCA_PARAM_TYPE_STRING == p->mbp_type) {
        if (NULL != p->mbp_default_value.stringval) {
            free(p->mbp_default_value.stringval);
        }
        if (p->mbp_file_value_set) {
            if (NULL != p->mbp_file_value.stringval) {
                free(p->mbp_file_value.stringval);
            }
            if (NULL != p->mbp_source_file) {
                free(p->mbp_source_file);
            }
        }
        if (p->mbp_override_value_set &&
            NULL != p->mbp_override_value.stringval) {
            free(p->mbp_override_value.stringval);
        }
    }

    /* Destroy any synonyms that are on the list */
    if (NULL != p->mbp_synonyms) {
        for (item = parsec_list_pop_front(p->mbp_synonyms);
             NULL != item; item = parsec_list_pop_front(p->mbp_synonyms)) {
            OBJ_RELEASE(item);
        }
        OBJ_RELEASE(p->mbp_synonyms);
    }

    /* mark this parameter as invalid */
    p->mbp_type = PARSEC_MCA_PARAM_TYPE_MAX;

#if defined(PARSEC_DEBUG_PARANOID)
    /* Cheap trick to reset everything to NULL */
    param_constructor(p);
#endif
}


static void fv_constructor(parsec_mca_param_file_value_t *f)
{
    f->mbpfv_param = NULL;
    f->mbpfv_value = NULL;
    f->mbpfv_file = NULL;
}


static void fv_destructor(parsec_mca_param_file_value_t *f)
{
    if (NULL != f->mbpfv_param) {
        free(f->mbpfv_param);
    }
    if (NULL != f->mbpfv_value) {
        free(f->mbpfv_value);
    }
    if (NULL != f->mbpfv_file) {
        free(f->mbpfv_file);
    }
    fv_constructor(f);
}

static void info_constructor(parsec_mca_param_info_t *p)
{
    p->mbpp_index = -1;
    p->mbpp_type = PARSEC_MCA_PARAM_TYPE_MAX;

    p->mbpp_type_name = NULL;
    p->mbpp_component_name = NULL;
    p->mbpp_param_name = NULL;
    p->mbpp_full_name = NULL;

    p->mbpp_deprecated = false;

    p->mbpp_synonyms = NULL;
    p->mbpp_synonyms_len = 0;
    p->mbpp_synonym_parent = NULL;

    p->mbpp_read_only = false;
    p->mbpp_help_msg = NULL;
}

static void info_destructor(parsec_mca_param_info_t *p)
{
    if (NULL != p->mbpp_synonyms) {
        free(p->mbpp_synonyms);
    }
    /* No need to free any of the strings -- the pointers were copied
       by value from their corresponding parameter registration */

    info_constructor(p);
}

static void syn_info_constructor(parsec_syn_info_t *si)
{
    si->si_type_name = si->si_component_name = si->si_param_name =
        si->si_full_name = si->si_env_var_name = NULL;
    si->si_deprecated = si->si_deprecated_warning_shown = false;
}

static void syn_info_destructor(parsec_syn_info_t *si)
{
    if (NULL != si->si_type_name) {
        free(si->si_type_name);
    }
    if (NULL != si->si_component_name) {
        free(si->si_component_name);
    }
    if (NULL != si->si_param_name) {
        free(si->si_param_name);
    }
    if (NULL != si->si_full_name) {
        free(si->si_full_name);
    }
    if (NULL != si->si_env_var_name) {
        free(si->si_env_var_name);
    }

    syn_info_constructor(si);
}

int parsec_mca_param_find_int_name(const char *type,
                                  const char *param_name,
                                  char **env,
                                  int *current_value)
{
    char *tmp, *ptr;
    int len, i;
    int rc = PARSEC_ERR_NOT_FOUND;

    if (NULL == env) {
        return PARSEC_ERR_NOT_FOUND;
    }

    if ( asprintf(&tmp, "%s%s_%s", mca_prefix, type, param_name) == -1 ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    len = strlen(tmp);
    for (i=0; NULL != env[i]; i++) {
        if (0 == strncmp(tmp, env[i], len)) {
            ptr = strchr(env[i], '=');
            ptr++;
            *current_value = strtol(ptr, NULL, 10);
            rc = PARSEC_SUCCESS;
            break;
        }
    }
    free(tmp);
    return rc;
}

int parsec_mca_param_find_sizet_name(const char *type,
                                    const char *param_name,
                                    char **env,
                                    size_t *current_value)
{
    char *tmp, *ptr;
    int len, i;
    int rc = PARSEC_ERR_NOT_FOUND;

    if (NULL == env) {
        return PARSEC_ERR_NOT_FOUND;
    }

    if ( asprintf(&tmp, "%s%s_%s", mca_prefix, type, param_name) == -1 ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    len = strlen(tmp);
    for (i=0; NULL != env[i]; i++) {
        if (0 == strncmp(tmp, env[i], len)) {
            ptr = strchr(env[i], '=');
            ptr++;
            *current_value = (size_t)strtoll(ptr, NULL, 10);
            rc = PARSEC_SUCCESS;
            break;
        }
    }
    free(tmp);
    return rc;
}

int parsec_mca_param_find_string_name(const char *type,
                                     const char *param_name,
                                     char **env,
                                     char **current_value)
{
    char *tmp, *ptr;
    int len, i;
    int rc=PARSEC_ERR_NOT_FOUND;

    if (NULL == env) {
        return PARSEC_ERR_NOT_FOUND;
    }

    if ( asprintf(&tmp, "%s%s_%s", mca_prefix, type, param_name) == -1 ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    len = strlen(tmp);
    for (i=0; NULL != env[i]; i++) {
        if (0 == strncmp(tmp, env[i], len)) {
            ptr = strchr(env[i], '=');
            ptr++;
            *current_value = ptr;
            rc = PARSEC_SUCCESS;
            break;
        }
    }
    free(tmp);
    return rc;
}

static char *source_name(parsec_mca_param_source_t source,
                         const char *filename)
{
    char *ret;
    int rc;

    switch (source) {
    case MCA_PARAM_SOURCE_DEFAULT:
        return strdup("default value");

    case MCA_PARAM_SOURCE_ENV:
        return strdup("command line or environment variable");

    case MCA_PARAM_SOURCE_FILE:
        rc = asprintf(&ret, "file (%s)", filename);
        assert(0 < rc);
        return ret;

    case MCA_PARAM_SOURCE_OVERRIDE:
        return strdup("internal override");
    }
    (void)rc;  /* prevent the compiler form complaining */
    return strdup("unknown (!)");
}

int parsec_mca_param_check_exclusive_string(const char *type_a,
                                           const char *component_a,
                                           const char *param_a,
                                           const char *type_b,
                                           const char *component_b,
                                           const char *param_b)
{
    int i, ret;
    parsec_mca_param_source_t source_a, source_b;
    char *filename_a, *filename_b;

    i = parsec_mca_param_find(type_a, component_a, param_a);
    if (i < 0) {
        return PARSEC_ERR_NOT_FOUND;
    }
    ret = parsec_mca_param_lookup_source(i, &source_a, &filename_a);
    if (PARSEC_SUCCESS != ret) {
        return ret;
    }

    i = parsec_mca_param_find(type_b, component_b, param_b);
    if (i < 0) {
        return PARSEC_ERR_NOT_FOUND;
    }
    ret = parsec_mca_param_lookup_source(i, &source_b, &filename_b);
    if (PARSEC_SUCCESS != ret) {
        return ret;
    }

    if (MCA_PARAM_SOURCE_DEFAULT != source_a &&
        MCA_PARAM_SOURCE_DEFAULT != source_b) {
        size_t len;
        char *str_a, *str_b, *name_a, *name_b;

        /* Form cosmetic string names for A */
        str_a = source_name(source_a, filename_a);
        len = 5;
        if (NULL != type_a) len += strlen(type_a);
        if (NULL != component_a) len += strlen(component_a);
        if (NULL != param_a) len += strlen(param_a);
        name_a = calloc(1, len+1);
        if (NULL == name_a) {
            free(str_a);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        if (NULL != type_a) {
            strncat(name_a, type_a, len);
            strncat(name_a, "_", len);
        }
        if (NULL != component_a) strncat(name_a, component_a, len);
        strncat(name_a, "_", len);
        if (NULL != param_a) strncat(name_a, param_a, len);

        /* Form cosmetic string names for B */
        str_b = source_name(source_b, filename_b);
        len = 5;
        if (NULL != type_b) len += strlen(type_b);
        if (NULL != component_b) len += strlen(component_b);
        if (NULL != param_b) len += strlen(param_b);
        name_b = calloc(1, len+1);
        if (NULL == name_b) {
            free(str_a);
            free(name_a);
            free(str_b);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        if (NULL != type_b) {
            strncat(name_b, type_b, len);
            strncat(name_b, "_", len);
        }
        if (NULL != component_b) strncat(name_b, component_b, len);
        strncat(name_b, "_", len);
        if (NULL != param_b) strncat(name_b, param_b, len);

        /* Print it all out */
        parsec_show_help("help-mca-param.txt",
                       "mutually exclusive params",
                       true, name_a, str_a, name_b, str_b);

        /* Free the temp strings */
        free(str_a);
        free(name_a);
        free(str_b);
        free(name_b);
        return PARSEC_ERR_BAD_PARAM;
    }

    return PARSEC_SUCCESS;
}

int parsec_mca_var_env_name(const char *param_name,
                           char **env_name)
{
    int ret;

    assert (NULL != env_name);

    ret = asprintf(env_name, "%s%s", mca_prefix, param_name);
    if (0 > ret) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    return PARSEC_SUCCESS;
}

/*
 * Private variables - set some reasonable screen size defaults
 */
static const char *parsec_info_component_all = "all";
static const char *parsec_info_type_all = "all";
static int centerpoint = 24;
static int screen_width = 78;
static int parsec_info_pretty = 0;

/*
 * Prints the passed value in a pretty or parsable format.
 */
static void
parsec_info_out(const char *pretty_message, const char *plain_message, const char *value)
{
    int rc;
    size_t i, len, max_value_width;
    char *spaces = NULL;
    char *filler = NULL;
    char *pos, *v, savev, *ov;

#ifdef PARSEC_HAVE_ISATTY
    /* If we have isatty(), if this is not a tty, then disable
     * wrapping for grep-friendly behavior
     */
    if (0 == isatty(STDOUT_FILENO)) {
        screen_width = INT_MAX;
    }
#endif

#ifdef TIOCGWINSZ
    if (screen_width < INT_MAX) {
        struct winsize size;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, (char*) &size) >= 0) {
            screen_width = size.ws_col;
        }
    }
#endif

    /* Strip leading and trailing whitespace from the string value */
    ov = strdup(value);
    v = ov;
    len = strlen(v);
    if (isspace(v[0])) {
        char *newv;
        i = 0;
        while (isspace(v[i]) && i < len) {
            ++i;
        }
        newv = strdup(v + i);
        free(v);
        ov = newv;
        v = newv;
        len = strlen(v);
    }
    if (len > 0 && isspace(v[len - 1])) {
        i = len - 1;
        /* Note that i is size_t (unsigned), so we can't check for i
           >= 0.  But we don't need to, because if the value was all
           whitespace, stripping whitespace from the left (above)
           would have resulted in an empty string, and we wouldn't
           have gotten into this block. */
        while (isspace(v[i]) && i > 0) {
            --i;
        }
        v[i + 1] = '\0';
    }

    if (parsec_info_pretty && NULL != pretty_message) {
        if (centerpoint > (int)strlen(pretty_message)) {
            rc = asprintf(&spaces, "%*s", centerpoint -
                          (int)strlen(pretty_message), " ");
            assert(0 < rc);
        } else {
            spaces = strdup("");
#if PARSEC_ENABLE_DEBUG
            if (centerpoint < (int)strlen(pretty_message)) {
                parsec_show_help("help-mca-param.txt",
                                "developer warning: field too long", false,
                               pretty_message, centerpoint);
            }
#endif
        }
        max_value_width = screen_width - strlen(spaces) - strlen(pretty_message) - 2;
        if (0 < strlen(pretty_message)) {
            rc = asprintf(&filler, "%s%s: ", spaces, pretty_message);
            assert(0 < rc);
        } else {
            rc = asprintf(&filler, "%s  ", spaces);
            assert(0 < rc);
        }
        free(spaces);
        spaces = NULL;

        while (true) {
            if (strlen(v) < max_value_width) {
                printf("%s%s\n", filler, v);
                break;
            } else {
                rc = asprintf(&spaces, "%*s", centerpoint + 2, " ");
                (void)rc;
                /* Work backwards to find the first space before
                 * max_value_width
                 */
                savev = v[max_value_width];
                v[max_value_width] = '\0';
                pos = (char*)strrchr(v, (int)' ');
                v[max_value_width] = savev;
                if (NULL == pos) {
                    /* No space found < max_value_width.  Look for the first
                     * space after max_value_width.
                     */
                    pos = strchr(&v[max_value_width], ' ');

                    if (NULL == pos) {

                        /* There's just no spaces.  So just print it and be done. */

                        printf("%s%s\n", filler, v);
                        break;
                    } else {
                        *pos = '\0';
                        printf("%s%s\n", filler, v);
                        v = pos + 1;
                    }
                } else {
                    *pos = '\0';
                    printf("%s%s\n", filler, v);
                    v = pos + 1;
                }

                /* Reset for the next iteration */
                free(filler);
                filler = strdup(spaces);
                free(spaces);
                spaces = NULL;
            }
        }
        if (NULL != filler) {
            free(filler);
        }
        if (NULL != spaces) {
            free(spaces);
        }
    } else {
        if (NULL != plain_message && 0 < strlen(plain_message)) {
            printf("%s:%s\n", plain_message, value);
        } else {
            printf("  %s\n", value);
        }
    }
    if( NULL != ov )
        free(ov);
    (void)rc;
}

void parsec_mca_show_mca_params(parsec_list_t *info,
                               const char *type, const char *component,
                               bool pretty_print)
{
    parsec_list_item_t *i;
    parsec_mca_param_info_t *p;
    char *value_string, *empty = "";
    char *message, *content, *tmp;
    int value_int, j;
    parsec_mca_param_source_t source;
    char *src_file;
    int rc;

    parsec_info_pretty = pretty_print;
    for (i = PARSEC_LIST_ITERATOR_FIRST(info); i != PARSEC_LIST_ITERATOR_LAST(info);
         i = PARSEC_LIST_ITERATOR_NEXT(i)) {
        p = (parsec_mca_param_info_t*) i;

        if (NULL != p->mbpp_type_name && ((0 == strcmp(type, parsec_info_type_all) ||
                                           (0 == strcmp(type, p->mbpp_type_name))))) {
            if (0 == strcmp(component, parsec_info_component_all) ||
                NULL == p->mbpp_component_name ||
                (NULL != p->mbpp_component_name &&
                 0 == strcmp(component, p->mbpp_component_name))) {

                /* Find the source of the value */
                if (PARSEC_SUCCESS !=
                    parsec_mca_param_lookup_source(p->mbpp_index, &source, &src_file)) {
                    continue;
                }

                /* Make a char *for the default value.  Invoke a
                 * lookup because it may transform the char *("~/" ->
                 * "<home dir>/") or get the value from the
                 * environment, a file, etc.
                 */
                if (PARSEC_MCA_PARAM_TYPE_STRING == p->mbpp_type) {
                    parsec_mca_param_lookup_string(p->mbpp_index,
                                                 &value_string);

                    /* Can't let the char *be NULL because we
                     * assign it to a std::string, below
                     */
                    if (NULL == value_string) {
                        value_string = strdup(empty);
                    }
                } else {
                    parsec_mca_param_lookup_int(p->mbpp_index, &value_int);
                    rc = asprintf(&value_string, "%d", value_int);
                    assert(0 < rc);
                }

                /* Build up the strings for the output */

                if (pretty_print) {
                    rc = asprintf(&message, "MCA %s", p->mbpp_type_name);
                    assert(0 < rc);

                    /* Put in the real, full name (which may be
                     * different than the categorization).
                     */
                    rc = asprintf(&content, "%s \"%s\" (%s: <%s>, data source: ",
                                  p->mbpp_read_only ? "information" : "parameter",
                                  p->mbpp_full_name,
                                  p->mbpp_read_only ? "value" : "current value",
                                  (0 == strlen(value_string)) ? "none" : value_string);
                    assert(0 < rc);

                    /* Indicate where the param was set from */
                    switch(source) {
                        case MCA_PARAM_SOURCE_DEFAULT:
                            rc = asprintf(&tmp, "%sdefault value", content);
                            assert(0 < rc);
                            free(content);
                            content = tmp;
                            break;
                        case MCA_PARAM_SOURCE_ENV:
                            rc = asprintf(&tmp, "%senvironment or cmdline", content);
                            assert(0 < rc);
                            free(content);
                            content = tmp;
                            break;
                        case MCA_PARAM_SOURCE_FILE:
                            rc = asprintf(&tmp, "%sfile [%s]", content, src_file);
                            assert(0 < rc);
                            free(content);
                            content = tmp;
                            break;
                        case MCA_PARAM_SOURCE_OVERRIDE:
                            rc = asprintf(&tmp, "%sAPI override", content);
                            assert(0 < rc);
                            free(content);
                            content = tmp;
                            break;
                        default:
                            break;
                    }

                    /* Is this parameter deprecated? */
                    if (p->mbpp_deprecated) {
                        rc = asprintf(&tmp, "%s, deprecated", content);
                        assert(0 < rc);
                        free(content);
                        content = tmp;
                    }

                    /* Does this parameter have any synonyms? */
                    if (p->mbpp_synonyms_len > 0) {
                        rc = asprintf(&tmp, "%s, synonyms: ", content);
                        assert(0 < rc);
                        free(content);
                        content = tmp;
                        for (j = 0; j < p->mbpp_synonyms_len; ++j) {
                            if (j > 0) {
                                rc = asprintf(&tmp, "%s, %s", content, p->mbpp_synonyms[j]->mbpp_full_name);
                                assert(0 < rc);
                                free(content);
                                content = tmp;
                            } else {
                                rc = asprintf(&tmp, "%s%s", content, p->mbpp_synonyms[j]->mbpp_full_name);
                                assert(0 < rc);
                                free(content);
                                content = tmp;
                            }
                        }
                    }

                    /* Is this parameter a synonym of something else? */
                    else if (NULL != p->mbpp_synonym_parent) {
                        rc = asprintf(&tmp, "%s, synonym of: %s", content, p->mbpp_synonym_parent->mbpp_full_name);
                        assert(0 < rc);
                        free(content);
                        content = tmp;
                    }
                    rc = asprintf(&tmp, "%s)", content);
                    assert(0 < rc);
                    free(content);
                    content = tmp;
                    parsec_info_out(message, message, content);
                    free(message);
                    free(content);

                    /* If we have a help message, parsec_info_output it */
                    if (NULL != p->mbpp_help_msg) {
                        parsec_info_out("", "", p->mbpp_help_msg);
                    }
                } else {
                    /* build the message*/
                    rc = asprintf(&tmp, "mca:%s:%s:param:%s:", p->mbpp_type_name,
                             (NULL == p->mbpp_component_name) ? "base" : p->mbpp_component_name,
                             p->mbpp_full_name);
                    assert(0 < rc);

                    /* Output the value */
                    rc = asprintf(&message, "%svalue", tmp);
                    assert(0 < rc);
                    parsec_info_out(message, message, value_string);
                    free(message);

                    /* Indicate where the param was set from */
                    rc = asprintf(&message, "%sdata_source", tmp);
                    assert(0 < rc);
                    switch(source) {
                        case MCA_PARAM_SOURCE_DEFAULT:
                            content = strdup("default value");
                            break;
                        case MCA_PARAM_SOURCE_ENV:
                            content = strdup("environment-cmdline");
                            break;
                        case MCA_PARAM_SOURCE_FILE:
                            rc = asprintf(&content, "file: %s", src_file);
                            assert(0 < rc);
                            break;
                        case MCA_PARAM_SOURCE_OVERRIDE:
                            content = strdup("API override");
                            break;
                        default:
                            content = strdup("**INTERNAL ERROR: undefined source");
                            break;
                    }
                    parsec_info_out(message, message, content);
                    free(message);
                    free(content);

                    /* Output whether it's read only or writable */

                    rc = asprintf(&message, "%sstatus", tmp);
                    assert(0 < rc);
                    content = p->mbpp_read_only ? "read-only" : "writable";
                    parsec_info_out(message, message, content);
                    free(message);

                    /* If it has a help message, parsec_info_output that */

                    if (NULL != p->mbpp_help_msg) {
                        rc = asprintf(&message, "%shelp", tmp);
                        assert(0 < rc);
                        content = p->mbpp_help_msg;
                        parsec_info_out(message, message, content);
                        free(message);
                    }

                    /* Is this parameter deprecated? */
                    rc = asprintf(&message, "%sdeprecated", tmp);
                    assert(0 < rc);
                    content = p->mbpp_deprecated ? "yes" : "no";
                    parsec_info_out(message, message, content);
                    free(message);

                    /* Does this parameter have any synonyms? */
                    if (p->mbpp_synonyms_len > 0) {
                        for (j = 0; j < p->mbpp_synonyms_len; ++j) {
                            rc = asprintf(&message, "%ssynonym:name", tmp);
                            assert(0 < rc);
                            content = p->mbpp_synonyms[j]->mbpp_full_name;
                            parsec_info_out(message, message, content);
                            free(message);
                        }
                    }

                    /* Is this parameter a synonym of something else? */
                    else if (NULL != p->mbpp_synonym_parent) {
                        rc = asprintf(&message, "%ssynonym_of:name", tmp);
                        assert(0 < rc);
                        content = p->mbpp_synonym_parent->mbpp_full_name;
                        parsec_info_out(message, message, content);
                        free(message);
                    }
                }

                free(value_string);
            }
        }
    }
    (void)rc;
}

/*
 * Set an MCA parameter in the environment provided.
 * If environment is environ, the MCA parameter is into the global environment
 * and can be accessed when the parameter is actually registered.
 */
void parsec_setenv_mca_param( char *param, char *value, char ***env )
{
    char *name;
    (void) parsec_mca_var_env_name (param, &name);
    parsec_setenv(name, value, true, env);
    free(name);
}
