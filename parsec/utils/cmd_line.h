/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2012 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * General command line parsing facility for use throughout Open MPI.
 *
 * This scheme is inspired by the GNU getopt package.  Command line
 * options are registered.  Each option can have up to three different
 * matching tokens: a "short" name, a "single dash" name, and a "long"
 * name.  Each option can also take 0 or more arguments.  Finally,
 * each option can be repeated on the command line an arbitrary number
 * of times.
 *
 * The "short" name can only be a single letter, and will be found
 * after a single dash (e.g., "-a").  Multiple "short" names can be
 * combined into a single command line argument (e.g., "-abc" can be
 * equivalent to "-a -b -c").
 *
 * The "single dash" name is a multi-character name that only
 * requires a single dash.  This only exists to provide backwards
 * compatibility for some well-known command line options in prior
 * MPI implementations (e.g., "mpirun -np 3").  It should be used
 * sparingly.
 *
 * The "long" name is a multi-character name that is found after a
 * pair of dashes.  For example, "--some-option-name".
 *
 * A command line option is a combination of 1 or more of a short
 * name, single dash name, and a long name.  Any of the names may be
 * used on the command line; they are treated as synonyms.  For
 * example, say the following was used in for an executable named
 * "foo":
 *
 * \code
 * parsec_cmd_line_make_opt3(cmd, 'a', NULL, 'add', 1, "Add a user");
 * \endcode
 *
 * In this case, the following command lines are exactly equivalent:
 *
 * \verbatim
 * shell$ foo -a jsmith
 * shell$ foo --add jsmith
 * \endverbatim
 *
 * Note that this interface can also track multiple invocations of the
 * same option.  For example, the following is both legal and able to
 * be retrieved through this interface:
 *
 * \verbatim
 * shell$ foo -a jsmith -add bjones
 * \endverbatim
 *
 * The caller to this interface creates a command line handle
 * (parsec_cmd_line_t) with OBJ_NEW() and then uses it to register the
 * desired parameters via parsec_cmd_line_make_opt3(). Once all the
 * parameters have been registered, the user invokes
 * parsec_cmd_line_parse() with the command line handle and the argv/argc
 * pair to be parsed (typically the arguments from main()).  The parser
 * will examine the argv and find registered options and parameters.
 * It will stop parsing when it runs into an recognized string token or
 * the special "--" token.
 *
 * After the parse has occurred, various accessor functions can be
 * used to determine which options were selected, what parameters were
 * passed to them, etc.:
 *
 * - parsec_cmd_line_get_usage_msg() returns a string suitable for "help"
 *   kinds of messages.
 * - parsec_cmd_line_is_taken() returns a true or false indicating
 *   whether a given command line option was found on the command
 *   line.
 * - parsec_cmd_line_get_argc() returns the number of tokens parsed on
 *   the handle.
 * - parsec_cmd_line_get_argv() returns any particular string from the
 *   original argv.
 * - parsec_cmd_line_get_ninsts() returns the number of times a
 *   particular option was found on a command line.
 * - parsec_cmd_line_get_param() returns the Nth parameter in the Mth
 *   instance of a given parameter.
 * - parsec_cmd_line_get_tail() returns an array of tokens not parsed
 *   (i.e., if the parser ran into "--" or an unrecognized token).
 *
 * Note that a shortcut to creating a large number of options exists
 * -- one can make a table of parsec_cmd_line_init_t instances and the
 * table to parsec_cmd_line_create().  This creates an parsec_cmd_line_t
 * handle that is pre-seeded with all the options from the table
 * without the need to repeatedly invoke parsec_cmd_line_make_opt3() (or
 * equivalent).  This parsec_cmd_line_t instance is just like any other;
 * it is still possible to add more options via
 * parsec_cmd_line_make_opt3(), etc.
 */

#ifndef PARSEC_CMD_LINE_H
#define PARSEC_CMD_LINE_H

#include "parsec/class/parsec_object.h"
#include "parsec/class/list.h"
#include "parsec/sys/atomic.h"

#if defined(PARSEC_HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(PARSEC_HAVE_STDBOOL_H) */

BEGIN_C_DECLS
/**
 * \internal
 *
 * Main top-level handle.  This interface should not be used by users!
 */
struct parsec_cmd_line_t {
    /** Make this an OBJ handle */
    parsec_object_t super;

    /** Thread safety */
    parsec_atomic_lock_t lcl_mutex;

    /** List of cmd_line_option_t's (defined internally) */
    parsec_list_t lcl_options;

    /** Duplicate of argc from parsec_cmd_line_parse() */
    int lcl_argc;
    /** Duplicate of argv from parsec_cmd_line_parse() */
    char **lcl_argv;

    /** Parsed output; list of cmd_line_param_t's (defined internally) */
    parsec_list_t lcl_params;

    /** List of tail (unprocessed) arguments */
    int lcl_tail_argc;
    /** List of tail (unprocessed) arguments */
    char **lcl_tail_argv;
};
/**
 * \internal
 *
 * Convenience typedef
 */
typedef struct parsec_cmd_line_t parsec_cmd_line_t;

/**
 * Data types supported by the parser
 */
enum parsec_cmd_line_type_t {
    PARSEC_CMD_LINE_TYPE_NULL,
    PARSEC_CMD_LINE_TYPE_STRING,
    PARSEC_CMD_LINE_TYPE_INT,
    PARSEC_CMD_LINE_TYPE_SIZE_T,
    PARSEC_CMD_LINE_TYPE_BOOL,

    PARSEC_CMD_LINE_TYPE_MAX
};
/**
 * \internal
 *
 * Convenience typedef
 */
typedef enum parsec_cmd_line_type_t parsec_cmd_line_type_t;

/**
 * Datatype used to construct a command line handle; see
 * parsec_cmd_line_create().
 */
struct parsec_cmd_line_init_t {
    /** If want to set an MCA parameter, set its parameter name
        here. */
    const char *ocl_mca_param_name;

    /** "Short" name (i.e., "-X", where "X" is a single letter) */
    char ocl_cmd_short_name;
    /** "Single dash" name (i.e., "-foo").  The use of these are
        discouraged. */
    const char *ocl_cmd_single_dash_name;
    /** Long name (i.e., "--foo"). */
    const char *ocl_cmd_long_name;

    /** Number of parameters that this option takes */
    int ocl_num_params;

    /** If this parameter is encountered, its *first* parameter it
        saved here.  If the parameter is encountered again, the
        value is overwritten. */
    void *ocl_variable_dest;
    /** If an ocl_variable_dest is given, its datatype must be
        supplied as well. */
    parsec_cmd_line_type_t ocl_variable_type;

    /** Description of the command line option, to be used with
        parsec_cmd_line_get_usage_msg(). */
    const char *ocl_description;
};
/**
 * \internal
 *
 * Convenience typedef
 */
typedef struct parsec_cmd_line_init_t parsec_cmd_line_init_t;

/**
 * Top-level command line handle.
 *
 * This handle is used for accessing all command line functionality
 * (i.e., all parsec_cmd_line*() functions).  Multiple handles can be
 * created and simultaneously processed; each handle is independant
 * from others.
 *
 * The parsec_cmd_line_t handles are [simplisticly] thread safe;
 * processing is guaranteed to be mutually exclusive if multiple
 * threads invoke functions on the same handle at the same time --
 * access will be serialized in an unspecified order.
 *
 * Once finished, handles should be released with OBJ_RELEASE().  The
 * destructor for parsec_cmd_line_t handles will free all memory
 * associated with the handle.
 */
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_cmd_line_t);

/**
 * Make a command line handle from a table of initializers.
 *
 * @param cmd PARSEC command line handle.
 * @param table Table of parsec_cmd_line_init_t instances for all
 * the options to be included in the resulting command line
 * handler.
 *
 * @retval PARSEC_SUCCESS Upon success.
 *
 * This function takes a table of parsec_cmd_line_init_t instances
 * to pre-seed an PARSEC command line handle.  The last instance in
 * the table must have '\0' for the short name and NULL for the
 * single-dash and long names.  The handle is expected to have
 * been OBJ_NEW'ed or OBJ_CONSTRUCT'ed already.
 *
 * Upon return, the command line handle is just like any other.  A
 * sample using this syntax:
 *
 * \code
 * parsec_cmd_line_init_t cmd_line_init[] = {
 *    { NULL, NULL, NULL, 'h', NULL, "help", 0,
 *      &orterun_globals.help, PARSEC_CMD_LINE_TYPE_BOOL,
 *      "This help message" },
 *
 *    { NULL, NULL, NULL, '\0', NULL, "wd", 1,
 *      &orterun_globals.wd, PARSEC_CMD_LINE_TYPE_STRING,
 *      "Set the working directory of the started processes" },
 *
 *    { NULL, NULL, NULL, '\0', NULL, NULL, 0,
 *      NULL, PARSEC_CMD_LINE_TYPE_NULL, NULL }
 * };
 * \endcode
 */
PARSEC_DECLSPEC int parsec_cmd_line_create(parsec_cmd_line_t *cmd,
                                         parsec_cmd_line_init_t *table);

/**
 * Create a command line option.
 *
 * @param cmd PARSEC command line handle.
 * @param entry Command line entry to add to the command line.
 *
 * @retval PARSEC_SUCCESS Upon success.
 *
 */
PARSEC_DECLSPEC int parsec_cmd_line_make_opt_mca(parsec_cmd_line_t *cmd,
                                               parsec_cmd_line_init_t entry);

/**
 * Create a command line option.
 *
 * @param cmd PARSEC command line handle.
 * @param short_name "Short" name of the command line option.
 * @param sd_name "Single dash" name of the command line option.
 * @param long_name "Long" name of the command line option.
 * @param num_params How many parameters this option takes.
 * @param dest Short string description of this option.
 *
 * @retval PARSEC_ERR_OUT_OF_RESOURCE If out of memory.
 * @retval PARSEC_ERR_BAD_PARAM If bad parameters passed.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * Adds a command line option to the list of options that an PARSEC
 * command line handle will accept.  The short_name may take the
 * special value '\0' to not have a short name.  Likewise, the
 * sd_name and long_name may take the special value NULL to not have
 * a single dash or long name, respectively.  However, one of the
 * three must have a name.
 *
 * num_params indicates how many parameters this option takes.  It
 * must be greater than or equal to 0.
 *
 * Finally, desc is a short string description of this option.  It is
 * used to generate the output from parsec_cmd_line_get_usage_msg().
 *
 */
PARSEC_DECLSPEC int parsec_cmd_line_make_opt3(parsec_cmd_line_t *cmd,
                                            char short_name,
                                            const char *sd_name,
                                            const char *long_name,
                                            int num_params,
                                            const char *desc);

/**
 * Parse a command line according to a pre-built PARSEC command line
 * handle.
 *
 * @param cmd PARSEC command line handle.
 * @param ignore_unknown Whether to print an error message upon
 * finding an unknown token or not
 * @param argc Length of the argv array.
 * @param argv Array of strings from the command line.
 *
 * @retval PARSEC_SUCCESS Upon success.
 * @retval PARSEC_ERR_SILENT If an error message was printed.  This
 * value will only be returned if the command line was not
 * successfully parsed.
 *
 * Parse a series of command line tokens according to the option
 * descriptions from a PARSEC command line handle.  The PARSEC command line
 * handle can then be queried to see what options were used, what
 * their parameters were, etc.
 *
 * If an unknown token is found in the command line (i.e., a token
 * that is not a parameter or a registered option), the parsing will
 * stop (see below).  If ignore_unknown is false, an error message
 * is displayed.  If ignore_unknown is true, the error message is
 * not displayed.
 *
 * Error messages are always displayed regardless of the value
 * of ignore_unknown (to stderr, and PARSEC_ERR_SILENT is
 * returned) if:
 *
 * 1. A token was encountered that required N parameters, but <N
 * parameters were found (e.g., "cmd --param foo", but --param was
 * registered to require 2 option tokens).
 *
 * 2. An unknown token beginning with "-" is encountered.  For
 * example, if "--fo" is specified, and no "fo" option is
 * registered (e.g., perhaps the user meant to type "--foo"), an
 * error message is always printed, UNLESS this unknown token
 * happens after a "--" token (see below).
 *
 * The contents of argc and argv are not changed during parsing.
 * argv[0] is assumed to be the executable name, and is ignored during
 * parsing, except when printing error messages.
 *
 * Parsing will stop in the following conditions:
 *
 * - all argv tokens are processed
 * - the token "--" is found
 * - an unrecognized token is found
 * - a parameter registered with an integer type option finds a
 *   non-integer option token
 * - a parameted registered N option tokens, but finds less then
 *   <N tokens available
 *
 * Upon any of these conditions, any remaining tokens will be placed
 * in the "tail" (and therefore not examined by the parser),
 * regardless of the value of ignore_unknown.  The set of tail
 * tokens is available from the parsec_cmd_line_get_tail() function.
 *
 * Note that "--" is ignored if it is found in the middle an expected
 * number of arguments.  For example, if "--foo" is expected to have 3
 * arguments, and the command line is:
 *
 * executable --foo a b -- other arguments
 *
 * This will result in an error, because "--" will be parsed as the
 * third parameter to the first instance of "foo", and "other" will be
 * an unrecognized option.
 *
 * Note that -- can be used to allow unknown tokens that begin
 * with "-".  For example, if a user wants to mpirun an executable
 * named "-my-mpi-program", the "usual" way:
 *
 *   mpirun -my-mpi-program
 *
 * will cause an error, because mpirun won't find single-letter
 * options registered for some/all of those letters.  But two
 * workarounds are possible:
 *
 *   mpirun -- -my-mpi-program
 * or
 *   mpirun ./-my-mpi-program
 *
 * Finally, note that invoking this function multiple times on
 * different sets of argv tokens is safe, but will erase any
 * previous parsing results.
 */
PARSEC_DECLSPEC int parsec_cmd_line_parse(parsec_cmd_line_t *cmd,
                                        bool ignore_unknown,
                                        int argc, char **argv);

/**
 * Return a consolidated "usage" message for a PARSEC command line handle.
 *
 * @param cmd PARSEC command line handle.
 *
 * @retval str Usage message.
 *
 * Returns a formatted string suitable for printing that lists the
 * expected usage message and a short description of each option on
 * the PARSEC command line handle.  Options that passed a NULL
 * description to parsec_cmd_line_make_opt3() will not be included in the
 * display (to allow for undocumented options).
 *
 * This function is typically only invoked internally by the
 * parsec_show_help() function.
 *
 * This function should probably be fixed up to produce prettier
 * output.
 *
 * The returned string must be freed by the caller.
 */
PARSEC_DECLSPEC char *parsec_cmd_line_get_usage_msg(parsec_cmd_line_t *cmd);

/**
 * Test if a given option was taken on the parsed command line.
 *
 * @param cmd PARSEC command line handle.
 * @param opt Short or long name of the option to check for.
 *
 * @retval true If the command line option was found during
 * parsec_cmd_line_parse().
 *
 * @retval false If the command line option was not found during
 * parsec_cmd_line_parse(), or parsec_cmd_line_parse() was not invoked on
 * this handle.
 *
 * This function should only be called after parsec_cmd_line_parse().
 *
 * The function will return true if the option matching opt was found
 * (either by its short or long name) during token parsing.
 * Otherwise, it will return false.
 */
PARSEC_DECLSPEC bool parsec_cmd_line_is_taken(parsec_cmd_line_t *cmd, const char *opt);

/**
 * Return the number of arguments parsed on a PARSEC command line handle.
 *
 * @param cmd A pointer to the PARSEC command line handle.
 *
 * @retval PARSEC_ERROR If cmd is NULL.
 * @retval argc Number of arguments previously added to the handle.
 *
 * Arguments are added to the handle via the parsec_cmd_line_parse()
 * function.
 */
PARSEC_DECLSPEC int parsec_cmd_line_get_argc(parsec_cmd_line_t *cmd);

/**
 * Return a string argument parsed on a PARSEC command line handle.
 *
 * @param cmd A pointer to the PARSEC command line handle.
 * @param index The nth argument from the command line (0 is
 * argv[0], etc.).
 *
 * @retval NULL If cmd is NULL or index is invalid
 * @retval argument String of original argv[index]
 *
 * This function returns a single token from the arguments parsed
 * on this handle.  Arguments are added bia the
 * parsec_cmd_line_parse() function.
 *
 * What is returned is a pointer to the actual string that is on
 * the handle; it should not be modified or freed.
 */
PARSEC_DECLSPEC char *parsec_cmd_line_get_argv(parsec_cmd_line_t *cmd,
                                             int index);

/**
 * Return the number of instances of an option found during parsing.
 *
 * @param cmd PARSEC command line handle.
 * @param opt Short or long name of the option to check for.
 *
 * @retval num Number of instances (to include 0) of a given potion
 * found during parsec_cmd_line_parse().
 *
 * @retval PARSEC_ERR If the command line option was not found during
 * parsec_cmd_line_parse(), or parsec_cmd_line_parse() was not invoked on
 * this handle.
 *
 * This function should only be called after parsec_cmd_line_parse().
 *
 * The function will return the number of instances of a given option
 * (either by its short or long name) -- to include 0 -- or PARSEC_ERR if
 * either the option was not specified as part of the PARSEC command line
 * handle, or parsec_cmd_line_parse() was not invoked on this handle.
 */
PARSEC_DECLSPEC int parsec_cmd_line_get_ninsts(parsec_cmd_line_t *cmd, const char *opt);

/**
 * Return a specific parameter for a specific instance of a option
 * from the parsed command line.
 *
 * @param cmd PARSEC command line handle.
 * @param opt Short or long name of the option to check for.
 * @param instance_num Instance number of the option to query.
 * @param param_num Which parameter to return.
 *
 * @retval param String of the parameter.
 * @retval NULL If any of the input values are invalid.
 *
 * This function should only be called after parsec_cmd_line_parse().
 *
 * This function returns the Nth parameter for the Ith instance of a
 * given option on the parsed command line (both N and I are
 * zero-indexed).  For example, on the command line:
 *
 * executable --foo bar1 bar2 --foo bar3 bar4
 *
 * The call to parsec_cmd_line_get_param(cmd, "foo", 1, 1) would return
 * "bar4".  parsec_cmd_line_get_param(cmd, "bar", 0, 0) would return
 * NULL, as would parsec_cmd_line_get_param(cmd, "foo", 2, 2);
 *
 * The returned string should \em not be modified or freed by the
 * caller.
 */
PARSEC_DECLSPEC char *parsec_cmd_line_get_param(parsec_cmd_line_t *cmd,
                                              const char *opt,
                                              int instance_num,
                                              int param_num);

/**
 * Return the entire "tail" of unprocessed argv from a PARSEC
 * command line handle.
 *
 * @param cmd A pointer to the PARSEC command line handle.
 * @param tailc Pointer to the output length of the null-terminated
 * tail argv array.
 * @param tailv Pointer to the output null-terminated argv of all
 * unprocessed arguments from the command line.
 *
 * @retval PARSEC_ERROR If cmd is NULL or otherwise invalid.
 * @retval PARSEC_SUCCESS Upon success.
 *
 * The "tail" is all the arguments on the command line that were
 * not processed for some reason.  Reasons for not processing
 * arguments include:
 *
 * \sa The argument was not recognized
 * \sa The argument "--" was seen, and therefore all arguments
 * following it were not processed
 *
 * The output tailc parameter will be filled in with the integer
 * length of the null-terminated tailv array (length including the
 * final NULL entry).  The output tailv parameter will be a copy
 * of the tail parameters, and must be freed (likely with a call
 * to parsec_argv_free()) by the caller.
 */
PARSEC_DECLSPEC int parsec_cmd_line_get_tail(parsec_cmd_line_t *cmd, int *tailc,
                                           char ***tailv);

END_C_DECLS


#endif /* PARSEC_CMD_LINE_H */
