
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
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
 * In general, these functions are intended to be used as follows:
 *
 * - Creating MCA parameters
 * -# Register a parameter, get an index back
 * - Using MCA parameters
 * -# Lookup a "normal" parameter value on a specific index, or
 * -# Lookup an attribute parameter on a specific index
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
 *   read only once -- when the first time any parsec_mca_param_t function is
 *   invoked).
 * - If nothing else was found, use the parameter's default value.
 *
 * Note that there is a second header file (mca_param_internal.h)
 * that contains several internal type delcarations for the parameter
 * system.  The internal file is only used within the parameter system
 * itself; it should not be required by any other Open MPI entities.
 */

#ifndef PARSEC_MCA_PARAM_H
#define PARSEC_MCA_PARAM_H

#include "parsec/class/parsec_object.h"
#if defined(PARSEC_HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(PARSEC_HAVE_STDBOOL_H) */
#include "parsec/class/list.h"

#include "parsec/constants.h"

/**
 * The types of MCA parameters.
 */
typedef enum {
    /** Parameter not existent or other error related to the MCA params. */
    PARSEC_MCA_PARAM_ERROR = -1,
    /** The parameter is of type signed integer. */
    PARSEC_MCA_PARAM_TYPE_INT,
    /** The parameter is of type size_t. */
    PARSEC_MCA_PARAM_TYPE_SIZET,
    /** The parameter is of type string. */
    PARSEC_MCA_PARAM_TYPE_STRING,

    /** Maximum parameter type. */
    PARSEC_MCA_PARAM_TYPE_MAX
} parsec_mca_param_type_t;


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
} parsec_mca_param_source_t;


/**
 * Struct for holding name/type info.  Used in mca_param_dump(),
 * below.
 */
struct parsec_mca_param_info_t {
    /** So that we can be in a list */
    parsec_list_item_t super;

    /** Index of this parameter */
    int mbpp_index;
    /** Enum indicating the back-end type of the parameter */
    parsec_mca_param_type_t mbpp_type;

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
    struct parsec_mca_param_info_t **mbpp_synonyms;
    /** Length of mbpp_synonyms array */
    int mbpp_synonyms_len;
    /** Back pointer to another parsec_mca_param_info_t that *this*
        param is a synonym of (or NULL) */
    struct parsec_mca_param_info_t *mbpp_synonym_parent;

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
typedef struct parsec_mca_param_info_t parsec_mca_param_info_t;

/*
 * Global functions for MCA
 */

BEGIN_C_DECLS

/**
 * Make a real object for the info
 */
PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_mca_param_info_t);

/**
 * Initialize the MCA parameter system.
 *
 * @retval PARSEC_SUCCESS
 *
 * This function initalizes the MCA parameter system.  It is
 * invoked internally (by parsec_mca_open()) and is only documented
 * here for completeness.
 */
PARSEC_DECLSPEC int parsec_mca_param_init(void);

/**
 * Recache the MCA param files
 *
 * @retval PARSEC_SUCCESS
 *
 */
PARSEC_DECLSPEC int parsec_mca_param_recache_files(void);

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
 * @retval PARSEC_ERROR Upon failure to register the parameter.
 * @retval index Index value that can be used with
 * parsec_mca_param_lookup_int() to retrieve the value of the
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
 * Note that if you use parsec_mca_param_find() to lookup the index
 * of the registered parameter, the "component" argument should be
 * NULL (because it is not specified in this registration
 * function, and is therefore registered with a NULL value).
 */
 PARSEC_DECLSPEC int
 parsec_mca_param_reg_int_name(const char *type,
                              const char *param_name,
                              const char *help_msg,
                              bool internal,
                              bool read_only,
                              int default_value,
                              int *current_value);

/**
 * Register a size_t MCA parameter that is not associated with a
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
 * @retval PARSEC_ERROR Upon failure to register the parameter.
 * @retval index Index value that can be used with
 * parsec_mca_param_lookup_size_t() to retrieve the value of the
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
 * Note that if you use parsec_mca_param_find() to lookup the index
 * of the registered parameter, the "component" argument should be
 * NULL (because it is not specified in this registration
 * function, and is therefore registered with a NULL value).
 */
 PARSEC_DECLSPEC int
 parsec_mca_param_reg_sizet_name(const char *type,
                                const char *param_name,
                                const char *help_msg,
                                bool internal,
                                bool read_only,
                                size_t default_value,
                                size_t *current_value);

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
 * @retval PARSEC_ERROR Upon failure to register the parameter.
 * @retval index Index value that can be used with
 * parsec_mca_param_lookup_string() to retrieve the value of the
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
 * Note that if you use parsec_mca_param_find() to lookup the index
 * of the registered parameter, the "component" argument should be
 * NULL (because it is not specified in this registration
 * function, and is therefore registered with a NULL value).
 */
PARSEC_DECLSPEC int
parsec_mca_param_reg_string_name(const char *type,
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
 * @param syn_type [in] Although this synonym is not associated with
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
PARSEC_DECLSPEC int
parsec_mca_param_reg_syn_name(int original_index,
                             const char *syn_type,
                             const char *syn_param_name,
                             bool deprecated);

/**
 * Deregister a MCA parameter
 *
 * @param index Index returned from mca_param_register_init()
 *
 */
PARSEC_DECLSPEC int parsec_mca_param_deregister(int index);

/**
 * Look up an integer MCA parameter.
 *
 * @param index Index previous returned from
 * mca_param_reg_int().
 * @param value Pointer to int where the parameter value will be
 * stored.
 *
 * @return PARSEC_ERROR Upon failure.  The contents of value are
 * undefined.
 * @return PARSEC_SUCCESS Upon success.  value will be filled with the
 * parameter's current value.
 *
 * The value of a specific MCA parameter can be looked up using the
 * return value from mca_param_reg_int().
 */
PARSEC_DECLSPEC int parsec_mca_param_lookup_int(int index, int *value);

/**
 * Look up a size_t MCA parameter.
 *
 * @param index Index previous returned from
 * mca_param_reg_sizet().
 * @param value Pointer to int where the parameter value will be
 * stored.
 *
 * @return PARSEC_ERROR Upon failure.  The contents of value are
 * undefined.
 * @return PARSEC_SUCCESS Upon success.  value will be filled with the
 * parameter's current value.
 *
 * The value of a specific MCA parameter can be looked up using the
 * return value from mca_param_reg_sizet().
 */
PARSEC_DECLSPEC int parsec_mca_param_lookup_sizet(int index, size_t *value);

/**
 * Look up a string MCA parameter.
 *
 * @param index Index previous returned from
 * mca_param_reg_string().
 * @param value Pointer to (char *) where the parameter value will be
 * stored.
 *
 * @return PARSEC_ERROR Upon failure.  The contents of value are
 * undefined.
 * @return PARSEC_SUCCESS Upon success.  value will be filled with the
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
PARSEC_DECLSPEC int parsec_mca_param_lookup_string(int index, char **value);

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
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * This function looks up to see where the value of an MCA
 * parameter came from.
 */
PARSEC_DECLSPEC int
parsec_mca_param_lookup_source(int index,
                              parsec_mca_param_source_t *source,
                              char **source_file);

/**
 * Sets an "override" value for an integer MCA parameter.
 *
 * @param index [in] Index of MCA parameter to set
 * @param value [in] The integer value to set
 *
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval PARSEC_SUCCESS Upon success.
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
PARSEC_DECLSPEC int parsec_mca_param_set_int(int index, int value);

/**
 * Sets an "override" value for a size_t MCA parameter.
 *
 * @param index [in] Index of MCA parameter to set
 * @param value [in] The size_t value to set
 *
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * This function sets a size_t value on the MCA parameter
 * indicated by the index value index.  This value will be used in
 * lieu of any other value from any other MCA source (environment
 * variable, file, etc.) until the value is unset with
 * mca_param_unset().
 *
 * This function may be invoked multiple times; each time, the
 * last "set" value is replaced with the newest value.
 */
PARSEC_DECLSPEC int parsec_mca_param_set_sizet(int index, size_t value);

/**
 * Sets an "override" value for an string MCA parameter.
 *
 * @param index [in] Index of MCA parameter to set
 * @param value [in] The string value to set
 *
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * This function sets a string value on the MCA parameter
 * indicated by the index value index.  This value will be used in
 * lieu of any other value from any other MCA source (environment
 * variable, file, etc.) until the value is unset with
 * parsec_mca_param_unset().
 *
 * The string is copied by value; the string "value" parameter
 * does not become "owned" by the parameter subsystem.
 *
 * This function may be invoked multiple times; each time, the
 * last "set" value is replaced with the newest value (the old
 * value is discarded).
 */
PARSEC_DECLSPEC int parsec_mca_param_set_string(int index, char *value);

/**
 * Unset a parameter that was previously set by
 * parsec_mca_param_set_*().
 *
 * @param index [in] Index of MCA parameter to set
 *
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * Resets previous value that was set (if any) on the given MCA
 * parameter.
 */
PARSEC_DECLSPEC int parsec_mca_param_unset(int index);

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
PARSEC_DECLSPEC char *parsec_mca_param_env_var(const char *param_name);

/**
 * Find the index for an MCA parameter based on its names.
 *
 * @param type Name of the type containing the parameter.
 * @param component Name of the component containing the parameter.
 * @param param Name of the parameter.
 *
 * @retval PARSEC_ERROR If the parameter was not found.
 * @retval index If the parameter was found.
 *
 * It is not always convenient to widely propagate a parameter's index
 * value, or it may be necessary to look up the parameter from a
 * different component -- where it is not possible to have the return
 * value from mca_param_reg_*().
 * This function can be used to look up the index of any registered
 * parameter.  The returned index can be used with
 * parsec_mca_param_lookup_*().
 */
PARSEC_DECLSPEC int
parsec_mca_param_find(const char *type,
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
 * @retval PARSEC_ERROR If the parameter was not found.
 *
 * Look for a specific MCA parameter in an environment and return its value
 */
PARSEC_DECLSPEC int
parsec_mca_param_find_int_name(const char *type,
                              const char *param_name,
                              char **env,
                              int *current_value);

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
 * @retval PARSEC_ERROR If the parameter was not found.
 *
 * Look for a specific MCA parameter in an environment and return its value
 */
PARSEC_DECLSPEC int
parsec_mca_param_find_sizet_name(const char *type,
                                const char *param_name,
                                char **env,
                                size_t *current_value);

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
 * @retval PARSEC_ERROR If the parameter was not found.
 *
 * Look for a specific MCA parameter in an environment and return its value
 */
PARSEC_DECLSPEC int
parsec_mca_param_find_string_name(const char *type,
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
 * This function will print an parsec_show_help() message and return
 * PARSEC_ERR_BAD_PARAM if it finds that the two parameters both have
 * value sources that are not MCA_PARAM_SOURCE_DEFAULT.  This
 * means that both parameters have been set by the user (i.e., they're
 * not default values).
 *
 * Note that parsec_show_help() allows itself to be hooked, so if this
 * happens after the aggregated orte_show_help() system is
 * initialized, the messages will be aggregated (w00t).
 *
 * @returns PARSEC_ERR_BAD_PARAM if the two parameters have sources that
 * are not MCA_PARAM_SOURCE_DEFAULT.
 * @returns PARSEC_SUCCESS otherwise.
 */
PARSEC_DECLSPEC int
parsec_mca_param_check_exclusive_string(const char *type_a,
                                       const char *component_a,
                                       const char *param_a,
                                       const char *type_b,
                                       const char *component_b,
                                       const char *param_b);

/**
 * Set the "internal" flag on an MCA parameter to true or false.
 *
 * @param index [in] Index previous returned from
 * mca_param_reg_*().
 * @param internal [in] Boolean indicating whether the MCA
 * parameter is internal (private) or public.
 *
 * @returns PARSEC_SUCCESS If it can find the parameter to reset
 * @returns PARSEC_ERROR Otherwise
 *
 * "Internal" MCA parameters are ones that are not intentended to
 * be seen or modified by users or user applications.  These
 * include values that are set at run time, such as TCP ports, IP
 * addresses, etc.  By setting the "internal" flag, internal MCA
 * parameters are not displayed during the output of ompi_info and
 * MPI_INIT (at least, they're not displayed by default), thus
 * keeping them away from prying user eyes.
 */
PARSEC_DECLSPEC int
parsec_mca_param_set_internal(int index, bool internal);

/**
 * Obtain a list of all the MCA parameters currently defined as
 * well as their types.
 *
 * @param info [out] An parsec_list_t of parsec_mca_param_info_t
 * instances.
 * @param internal [in] Whether to include the internal parameters
 * or not.
 *
 * @retval PARSEC_SUCCESS Upon success.
 * @retval PARSEC_ERROR Upon failure.
 *
 * This function is used to obtain a list of all the currently
 * registered MCA parameters along with their associated types
 * (currently: string or integer).  The results from this function
 * can be used to repeatedly invoke parsec_mca_param_lookup_*()
 * to obtain a comprehensive list of all MCA parameters and
 * their current values.
 *
 * Releasing the list, and all the items in the list, is a
 * relatively complicated process.  Use the companion function
 * mca_param_dump_release() when finished with the returned
 * info list to release all associated memory.
 */
PARSEC_DECLSPEC int
parsec_mca_param_dump(parsec_list_t **info, bool internal);

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
 * @retval PARSEC_SUCCESS Upon success.
 * @retval PARSEC_ERROR Upon failure.
 *
 * This function is similar to mca_param_dump() except that
 * its output is in terms of an argv-style array of key=value
 * strings, suitable for using in an environment.
 */
PARSEC_DECLSPEC int
parsec_mca_param_build_env(char ***env, int *num_env,
                          bool internal);

/**
 * Release the memory associated with the info list returned from
 * mca_param_dump().
 *
 * @param info [in/out] An parsec_list_t previously returned from
 * mca_param_dump().
 *
 * @retval PARSEC_SUCCESS Upon success.
 * @retval PARSEC_ERROR Upon failure.
 *
 * This function is intended to be used to free the info list
 * returned from mca_param_dump().  There are a bunch of
 * strings and other associated memory in the list making it
 * cumbersome for the caller to free it all properly.  Hence, once
 * the caller is finished with the info list, invoke this
 * function and all memory associated with the list will be freed.
 */
PARSEC_DECLSPEC int
parsec_mca_param_dump_release(parsec_list_t *info);

/**
 * Shut down the MCA parameter system (normally only invoked by the
 * MCA framework itself).
 *
 * @returns PARSEC_SUCCESS This function never fails.
 *
 * This function shuts down the MCA parameter repository and frees all
 * associated memory.  No other mca_param*() functions can be
 * invoked after this function.
 *
 * This function is normally only invoked by the MCA framework itself
 * when the process is shutting down (e.g., during MPI_FINALIZE).  It
 * is only documented here for completeness.
 */
PARSEC_DECLSPEC int
parsec_mca_param_finalize(void);

/**
 * Get the string name corresponding to the MCA variable
 * value in the environment.
 *
 * @param param_name Name of the type containing the variable.
 *
 * @retval string A string suitable for setenv() or appending to
 * an environ-style string array.
 * @retval NULL Upon failure.
 *
 * The string that is returned is owned by the caller; if
 * appropriate, it must be eventually freed by the caller.
 */
PARSEC_DECLSPEC int
parsec_mca_var_env_name(const char *param_name,
                       char **env_name);

PARSEC_DECLSPEC void
parsec_mca_show_mca_params(parsec_list_t *info,
                          const char *type, const char *component,
                          bool pretty_print);

/**
 * Set an MCA environment parameter.with a string value
 *
 * @param param Name of the type containing the variable.
 * @param value Value of the mca parameter to set.
 *
 * This function sets an MCA environment parameter in the global environment
 * of the application (i.e., environ) so that it cane be used as the default
 * value for the parameter when it is accessed by the PaRSEC engine. Thus, an
 * external application can register some parameters that will later be used by
 * the initialization in the engine.
 *
 */
PARSEC_DECLSPEC void
parsec_setenv_mca_param_string( char *param,
                                char *value);

/**
 * Set an MCA environment parameter.with an integer value
 *
 * @param param Name of the type containing the variable.
 * @param value Value of the mca parameter to set.
 *
 * This function sets an MCA environment parameter in the global environment
 * of the application (i.e., environ) so that it cane be used as the default
 * value for the parameter when it is accessed by the PaRSEC engine. Thus, an
 * external application can register some parameters that will later be used by
 * the initialization in the engine.
 *
 */
PARSEC_DECLSPEC void
parsec_setenv_mca_param_int( char *param,
                             int ivalue);
END_C_DECLS

#endif /* PARSEC_MCA_PARAM_H */
