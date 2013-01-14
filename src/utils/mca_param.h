
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2011 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/** @file
 * This file presents the MCA parameter interface.
 *
 * Note that there are two scopes for MCA parameters: "normal" and
 * attributes.  Specifically, all MCA parameters are "normal" -- some
 * are special and may also be found on attributes on communicators,
 * datatypes, or windows.
 *
 * In general, these functions are intended to be used as follows:
 *
 * - Creating MCA parameters
 * -# Register a parameter, get an index back
 * - Using MCA parameters
 * -# Lookup a "normal" parameter value on a specific index, or
 * -# Lookup an attribute parameter on a specific index and
 *    communicator / datatype / window.
 *
 * MCA parameters can be defined in multiple different places.  As
 * such, parameters are \em resolved to find their value.  The order
 * of resolution is as follows:
 *
 * - An "override" location that is only available to be set via the
 *   mca_param API.
 * - Look for an environment variable corresponding to the MCA
 *   parameter.
 * - See if a file contains the MCA parameter (MCA parameter files are
 *   read only once -- when the first time any dague_mca_param_t function is
 *   invoked).
 * - If nothing else was found, use the parameter's default value.
 *
 * Note that there is a second header file (mca_param_internal.h)
 * that contains several internal type delcarations for the parameter
 * system.  The internal file is only used within the parameter system
 * itself; it should not be required by any other Open MPI entities.
 */

#ifndef DAGUE_MCA_PARAM_H
#define DAGUE_MCA_PARAM_H

#include <dague_config.h>
#include <dague/class/dague_object.h>
#if defined(HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(HAVE_STDBOOL_H) */
#include "list.h"

/**
 * The types of MCA parameters.
 */
typedef enum {
    /** The parameter is of type integer. */
    DAGUE_MCA_PARAM_TYPE_INT,
    /** The parameter is of type string. */
    DAGUE_MCA_PARAM_TYPE_STRING,

    /** Maximum parameter type. */
    DAGUE_MCA_PARAM_TYPE_MAX
} dague_mca_param_type_t;


/**
 * Source of an MCA parameter's value
 */
typedef enum {
    /** The default value */
    MCA_PARAM_SOURCE_DEFAULT,
    /** The value came from the environment (or command line!) */
    MCA_PARAM_SOURCE_ENV,
    /** The value came from a file */
    MCA_PARAM_SOURCE_FILE,
    /** The value came a "set" API call */
    MCA_PARAM_SOURCE_OVERRIDE,

    /** Maximum source type */
    MCA_PARAM_SOURCE_MAX
} dague_mca_param_source_t;


/**
 * Struct for holding name/type info.  Used in mca_param_dump(),
 * below.
 */
struct dague_mca_param_info_t {
    /** So that we can be in a list */
    dague_list_item_t super;

    /** Index of this parameter */
    int mbpp_index;
    /** Enum indicating the back-end type of the parameter */
    dague_mca_param_type_t mbpp_type;

    /** String name of the type of this component */
    char *mbpp_type_name;
    /** String name of the component of the parameter */
    char *mbpp_component_name;
    /** String name of the parameter of the parameter */
    char *mbpp_param_name;
    /** Full, assembled parameter name */
    char *mbpp_full_name;

    /** Is this parameter deprecated? */
    bool mbpp_deprecated;

    /** Array of pointers of synonyms of this parameter */
    struct dague_mca_param_info_t **mbpp_synonyms;
    /** Length of mbpp_synonyms array */
    int mbpp_synonyms_len;
    /** Back pointer to another dague_mca_param_info_t that *this*
        param is a synonym of (or NULL) */
    struct dague_mca_param_info_t *mbpp_synonym_parent;

    /** Is this parameter internal? */
    bool mbpp_internal;
    /** Is this parameter changable? */
    bool mbpp_read_only;
    /** Help message associated with this parameter */
    char *mbpp_help_msg;
};
/**
 * Convenience typedef
 */
typedef struct dague_mca_param_info_t dague_mca_param_info_t;

/*
 * Global functions for MCA
 */

BEGIN_C_DECLS

/**
 * Make a real object for the info
 */
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_mca_param_info_t);

/**
 * Initialize the MCA parameter system.
 *
 * @retval DAGUE_SUCCESS
 *
 * This function initalizes the MCA parameter system.  It is
 * invoked internally (by mca_base_open()) and is only documented
 * here for completeness.
 */
DAGUE_DECLSPEC int dague_mca_param_init(void);

/**
 * Recache the MCA param files
 *
 * @retval DAGUE_SUCCESS
 *
 */
DAGUE_DECLSPEC int dague_mca_param_recache_files(void);

/**
 * Register an integer MCA parameter that is not associated with a
 * component.
 *
 * @param type [in] Although this parameter is not associated with
 * a component, it still must have a string type name that will
 * act as a prefix (string).
 * @param param_name [in] The name of the parameter being
 * registered (string).
 * @param help_msg [in] A string describing the use and valid
 * values of the parameter (string).
 * @param internal [in] Indicates whether the parameter is internal
 * (i.e., not to be shown to users) or not (bool).
 * @param read_only [in] Indicates whether the parameter value can
 * ever change (bool).
 * @param default_value [in] The value that is used for this
 * parameter if the user does not supply one.
 * @param current_value [out] After registering the parameter, look
 * up its current value and return it unless current_value is
 * NULL.
 *
 * @retval DAGUE_ERROR Upon failure to register the parameter.
 * @retval index Index value that can be used with
 * dague_mca_param_lookup_string() to retrieve the value of the
 * parameter.
 *
 * Note that the type should always be a framework or a level name
 * (e.g., "btl" or "mpi") -- it should not include the component
 * name, even if the component is the base of a framework.  Hence,
 * "btl_base" is not a valid type name.  Specifically, registering
 * a parameter with an unrecognized type is not an error, but
 * ompi_info has a hard-coded list of frameworks and levels;
 * parameters that have recongized types, although they can be
 * used by the user, will not be displayed by ompi_info.
 *
 * Note that if you use dague_mca_param_find() to lookup the index
 * of the registered parameter, the "component" argument should be
 * NULL (because it is not specified in this registration
 * function, and is therefore registered with a NULL value).
 */
 DAGUE_DECLSPEC int
 dague_mca_param_reg_int_name(const char *type,
                              const char *param_name,
                              const char *help_msg,
                              bool internal,
                              bool read_only,
                              int default_value,
                              int *current_value);

/**
 * Register a string MCA parameter that is not associated with a
 * component.
 *
 * @param type [in] Although this parameter is not associated with
 * a component, it still must have a string type name that will
 * act as a prefix (string).
 * @param param_name [in] The name of the parameter being
 * registered (string).
 * @param help_msg [in] A string describing the use and valid
 * values of the parameter (string).
 * @param internal [in] Indicates whether the parameter is internal
 * (i.e., not to be shown to users) or not (bool).
 * @param read_only [in] Indicates whether the parameter value can
 * ever change (bool).
 * @param default_value [in] The value that is used for this
 * parameter if the user does not supply one.
 * @param current_value [out] After registering the parameter, look
 * up its current value and return it unless current_value is
 * NULL.
 *
 * @retval DAGUE_ERROR Upon failure to register the parameter.
 * @retval index Index value that can be used with
 * dague_mca_param_lookup_string() to retrieve the value of the
 * parameter.
 *
 * Note that if a string value is read in from a file then it will
 * never be NULL. It will always have a value, even if that value is
 * the empty string.
 *
 * Note that the type should always be a framework or a level name
 * (e.g., "btl" or "mpi") -- it should not include the component
 * name, even if the component is the base of a framework.  Hence,
 * "btl_base" is not a valid type name.  Specifically, registering
 * a parameter with an unrecognized type is not an error, but
 * ompi_info has a hard-coded list of frameworks and levels;
 * parameters that have recongized types, although they can be
 * used by the user, will not be displayed by ompi_info.
 *
 * Note that if you use dague_mca_param_find() to lookup the index
 * of the registered parameter, the "component" argument should be
 * NULL (because it is not specified in this registration
 * function, and is therefore registered with a NULL value).
 */
DAGUE_DECLSPEC int
dague_mca_param_reg_string_name(const char *type,
                                const char *param_name,
                                const char *help_msg,
                                bool internal,
                                bool read_only,
                                const char *default_value,
                                char **current_value);

/**
 * Register an MCA parameter synonym that is not associated with a
 * component.
 *
 * @param original_index [in] The index of the original parameter to
 * create a synonym for.
 * @param type [in] Although this synonym is not associated with
 * a component, it still must have a string type name that will
 * act as a prefix (string).
 * @param syn_param_name [in] Parameter name of the synonym to be
 * created (string)
 * @param deprecated If true, a warning will be shown if this
 * synonym is used to set the parameter's value (unless the
 * warnings are silenced)
 *
 * Essentially the same as mca_param_reg_syn(), but using a
 * type name instead of a component.
 *
 * See mca_param_reg_int_name() for guidence on type string
 * values.
 */
DAGUE_DECLSPEC int
dague_mca_param_reg_syn_name(int orignal_index,
                             const char *syn_type,
                             const char *syn_param_name,
                             bool deprecated);

/**
 * Deregister a MCA parameter
 *
 * @param index Index returned from mca_param_register_init()
 *
 */
DAGUE_DECLSPEC int dague_mca_param_deregister(int index);

/**
 * Look up an integer MCA parameter.
 *
 * @param index Index previous returned from
 * mca_param_reg_int().
 * @param value Pointer to int where the parameter value will be
 * stored.
 *
 * @return DAGUE_ERROR Upon failure.  The contents of value are
 * undefined.
 * @return DAGUE_SUCCESS Upon success.  value will be filled with the
 * parameter's current value.
 *
 * The value of a specific MCA parameter can be looked up using the
 * return value from mca_param_reg_int().
 */
DAGUE_DECLSPEC int dague_mca_param_lookup_int(int index, int *value);

/**
 * Look up a string MCA parameter.
 *
 * @param index Index previous returned from
 * mca_param_reg_string().
 * @param value Pointer to (char *) where the parameter value will be
 * stored.
 *
 * @return DAGUE_ERROR Upon failure.  The contents of value are
 * undefined.
 * @return DAGUE_SUCCESS Upon success.  value will be filled with the
 * parameter's current value.
 *
 * Note that if a string value is read in from a file then it will
 * never be NULL. It will always have a value, even if that value is
 * the empty string.
 *
 * Strings returned in the \em value parameter should later be
 * free()'ed.
 *
 * The value of a specific MCA parameter can be looked up using the
 * return value from mca_param_reg_string().
 */
DAGUE_DECLSPEC int dague_mca_param_lookup_string(int index, char **value);

/**
 * Lookup the source of an MCA parameter's value
 *
 * @param index [in] Index of MCA parameter to set
 * @param source [out] Enum value indicating source
 * @param source_file [out] If value came from source, name of the
 * file that set it.  The caller should not modify or free this
 * string.  It is permissable to specify source_file==NULL if the
 * caller does not care to know the filename.
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 * @retval DAGUE_SUCCESS Upon success.
 *
 * This function looks up to see where the value of an MCA
 * parameter came from.
 */
DAGUE_DECLSPEC int
dague_mca_param_lookup_source(int index,
                              dague_mca_param_source_t *source,
                              char **source_file);

/**
 * Sets an "override" value for an integer MCA parameter.
 *
 * @param index [in] Index of MCA parameter to set
 * @param value [in] The integer value to set
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 * @retval DAGUE_SUCCESS Upon success.
 *
 * This function sets an integer value on the MCA parameter
 * indicated by the index value index.  This value will be used in
 * lieu of any other value from any other MCA source (environment
 * variable, file, etc.) until the value is unset with
 * mca_param_unset().
 *
 * This function may be invoked multiple times; each time, the
 * last "set" value is replaced with the newest value.
 */
DAGUE_DECLSPEC int dague_mca_param_set_int(int index, int value);

/**
 * Sets an "override" value for an string MCA parameter.
 *
 * @param index [in] Index of MCA parameter to set
 * @param value [in] The string value to set
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 * @retval DAGUE_SUCCESS Upon success.
 *
 * This function sets a string value on the MCA parameter
 * indicated by the index value index.  This value will be used in
 * lieu of any other value from any other MCA source (environment
 * variable, file, etc.) until the value is unset with
 * dague_mca_param_unset().
 *
 * The string is copied by value; the string "value" parameter
 * does not become "owned" by the parameter subsystem.
 *
 * This function may be invoked multiple times; each time, the
 * last "set" value is replaced with the newest value (the old
 * value is discarded).
 */
DAGUE_DECLSPEC int dague_mca_param_set_string(int index, char *value);

/**
 * Unset a parameter that was previously set by
 * dague_mca_param_set_int() or dague_mca_base_param_set_string().
 *
 * @param index [in] Index of MCA parameter to set
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 * @retval DAGUE_SUCCESS Upon success.
 *
 * Resets previous value that was set (if any) on the given MCA
 * parameter.
 */
DAGUE_DECLSPEC int dague_mca_param_unset(int index);

/**
 * Get the string name corresponding to the MCA parameter
 * value in the environment.
 *
 * @param param_name Name of the type containing the parameter.
 *
 * @retval string A string suitable for setenv() or appending to
 * an environ-style string array.
 * @retval NULL Upon failure.
 *
 * The string that is returned is owned by the caller; if
 * appropriate, it must be eventually freed by the caller.
 */
DAGUE_DECLSPEC char *dague_mca_param_env_var(const char *param_name);

/**
 * Find the index for an MCA parameter based on its names.
 *
 * @param type Name of the type containing the parameter.
 * @param component Name of the component containing the parameter.
 * @param param Name of the parameter.
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 * @retval index If the parameter was found.
 *
 * It is not always convenient to widely propagate a parameter's index
 * value, or it may be necessary to look up the parameter from a
 * different component -- where it is not possible to have the return
 * value from mca_param_reg_int() or mca_base_param_reg_string().
 * This function can be used to look up the index of any registered
 * parameter.  The returned index can be used with
 * mca_param_lookup_int() and mca_base_param_lookup_string().
 */
DAGUE_DECLSPEC int
dague_mca_param_find(const char *type,
                     const char *component,
                     const char *param);

/**
 * Find an MCA parameter (in an env array) that is not associated with a
 * component.
 *
 * @param type [in] Although this parameter is not associated with
 * a component, it still must have a string type name that will
 * act as a prefix (string).
 * @param param_name [in] The name of the parameter being
 * registered (string).
 * @param env [in] NULL-terminated list of strings (e.g., from an environment).
 * @param current_value [out] Return the current value (if found).
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 *
 * Look for a specific MCA parameter in an environment and return its value
 */
DAGUE_DECLSPEC int
dague_mca_param_find_int_name(const char *type,
                              const char *param_name,
                              char **env,
                              int *current_value);

/**
 * Find a string MCA parameter (in an env array) that is not associated with a
 * component.
 *
 * @param type [in] Although this parameter is not associated with
 * a component, it still must have a string type name that will
 * act as a prefix (string).
 * @param param_name [in] The name of the parameter being
 * registered (string).
 * @param env [in] NULL-terminated list of strings (e.g., from an environment).
 * @param current_value [out] Return the current value (if found).
 *
 * @retval DAGUE_ERROR If the parameter was not found.
 *
 * Look for a specific MCA parameter in an environment and return its value
 */
DAGUE_DECLSPEC int
dague_mca_param_find_string_name(const char *type,
                                 const char *param_name,
                                 char **env,
                                 char **current_value);

/**
 * Check that two MCA parameters were not both set to non-default
 * values.
 *
 * @param type_a [in] Framework name of parameter A (string).
 * @param component_a [in] Component name of parameter A (string).
 * @param param_a [in] Parameter name of parameter A (string.
 * @param type_b [in] Framework name of parameter A (string).
 * @param component_b [in] Component name of parameter A (string).
 * @param param_b [in] Parameter name of parameter A (string.
 *
 * This function is useful for checking that the user did not set both
 * of 2 mutually-exclusive MCA parameters.
 *
 * This function will print an dague_show_help() message and return
 * DAGUE_ERR_BAD_PARAM if it finds that the two parameters both have
 * value sources that are not MCA_PARAM_SOURCE_DEFAULT.  This
 * means that both parameters have been set by the user (i.e., they're
 * not default values).
 *
 * Note that dague_show_help() allows itself to be hooked, so if this
 * happens after the aggregated orte_show_help() system is
 * initialized, the messages will be aggregated (w00t).
 *
 * @returns DAGUE_ERR_BAD_PARAM if the two parameters have sources that
 * are not MCA_PARAM_SOURCE_DEFAULT.
 * @returns DAGUE_SUCCESS otherwise.
 */
DAGUE_DECLSPEC int
dague_mca_param_check_exclusive_string(const char *type_a,
                                       const char *component_a,
                                       const char *param_a,
                                       const char *type_b,
                                       const char *component_b,
                                       const char *param_b);

/**
 * Set the "internal" flag on an MCA parameter to true or false.
 *
 * @param index [in] Index previous returned from
 * mca_param_reg_string() or mca_base_param_reg_int().
 * @param internal [in] Boolean indicating whether the MCA
 * parameter is internal (private) or public.
 *
 * @returns DAGUE_SUCCESS If it can find the parameter to reset
 * @returns DAGUE_ERROR Otherwise
 *
 * "Internal" MCA parameters are ones that are not intentended to
 * be seen or modified by users or user applications.  These
 * include values that are set at run time, such as TCP ports, IP
 * addresses, etc.  By setting the "internal" flag, internal MCA
 * parameters are not displayed during the output of ompi_info and
 * MPI_INIT (at least, they're not displayed by default), thus
 * keeping them away from prying user eyes.
 */
DAGUE_DECLSPEC int
dague_mca_param_set_internal(int index, bool internal);

/**
 * Obtain a list of all the MCA parameters currently defined as
 * well as their types.
 *
 * @param info [out] An dague_list_t of dague_mca_param_info_t
 * instances.
 * @param internal [in] Whether to include the internal parameters
 * or not.
 *
 * @retval DAGUE_SUCCESS Upon success.
 * @retval DAGUE_ERROR Upon failure.
 *
 * This function is used to obtain a list of all the currently
 * registered MCA parameters along with their associated types
 * (currently: string or integer).  The results from this function
 * can be used to repeatedly invoke mca_param_lookup_int()
 * and/or mca_param_lookup_string() to obtain a comprehensive
 * list of all MCA parameters and their current values.
 *
 * Releasing the list, and all the items in the list, is a
 * relatively complicated process.  Use the companion function
 * mca_param_dump_release() when finished with the returned
 * info list to release all associated memory.
 */
DAGUE_DECLSPEC int
dague_mca_param_dump(dague_list_t **info, bool internal);

/**
 * Obtain a list of all the MCA parameters currently defined as
 * well as their types.
 *
 * @param env [out] A pointer to an argv-style array of key=value
 * strings, suitable for use in an environment
 * @param num_env [out] A pointer to an int, containing the length
 * of the env array (not including the final NULL entry).
 * @param internal [in] Whether to include the internal parameters
 * or not.
 *
 * @retval DAGUE_SUCCESS Upon success.
 * @retval DAGUE_ERROR Upon failure.
 *
 * This function is similar to mca_param_dump() except that
 * its output is in terms of an argv-style array of key=value
 * strings, suitable for using in an environment.
 */
DAGUE_DECLSPEC int
dague_mca_param_build_env(char ***env, int *num_env,
                          bool internal);

/**
 * Release the memory associated with the info list returned from
 * mca_param_dump().
 *
 * @param info [in/out] An dague_list_t previously returned from
 * mca_param_dump().
 *
 * @retval DAGUE_SUCCESS Upon success.
 * @retval DAGUE_ERROR Upon failure.
 *
 * This function is intended to be used to free the info list
 * returned from mca_param_dump().  There are a bunch of
 * strings and other associated memory in the list making it
 * cumbersome for the caller to free it all properly.  Hence, once
 * the caller is finished with the info list, invoke this
 * function and all memory associated with the list will be freed.
 */
DAGUE_DECLSPEC int
dague_mca_param_dump_release(dague_list_t *info);

/**
 * Shut down the MCA parameter system (normally only invoked by the
 * MCA framework itself).
 *
 * @returns DAGUE_SUCCESS This function never fails.
 *
 * This function shuts down the MCA parameter repository and frees all
 * associated memory.  No other mca_param*() functions can be
 * invoked after this function.
 *
 * This function is normally only invoked by the MCA framework itself
 * when the process is shutting down (e.g., during MPI_FINALIZE).  It
 * is only documented here for completeness.
 */
DAGUE_DECLSPEC int
dague_mca_param_finalize(void);

END_C_DECLS

#endif /* DAGUE_MCA_PARAM_H */
