#ifndef _jdf2c_utils_h
#define _jdf2c_utils_h

#include "string_arena.h"

typedef char *(*dumper_function_t)(void **elt, void *arg);

/**
 * UTIL_DUMP_LIST_FIELD:
 *    Iterate over the elements of a list, transforming each field element in a string using a parameter function,
 *    and concatenate all strings.
 *    The function has the prototype  field_t **e, void *a -> char *strelt
 *    The final string has the format
 *       before (prefix strelt separator)* (prefix strelt) after
 *  @param [IN] arena:         string arena to use to add strings elements to the final string
 *  @param [IN] structure_ptr: pointer to a structure that implement any list
 *  @param [IN] nextfield:     the name of a field pointing to the next structure pointer
 *  @param [IN] eltfield:      the name of a field pointing to an element to print
 *  @param [IN] fct:           a function that transforms a pointer to an element to a string of characters
 *  @param [IN] fctarg:        fixed argument of the function
 *  @param [IN] before:        string (of characters) representing what must appear before the list
 *  @param [IN] prefix:        string (of characters) representing what must appear before each element
 *  @param [IN] separator:     string (of characters) that will be put between each element, but not at the end
 *                             or before the first
 *  @param [IN] after:         string (of characters) that will be put at the end of the list, after the last
 *                             element
 *
 *  @return a string (of characters) written in arena with the list formed so.
 *
 *  If the function fct return NULL, the element is ignored
 *
 *  Example: to create the list of expressions that is a parameter call, use
 *    UTIL_DUMP_LIST_FIELD(sa, jdf->functions->predicates, next, expr, dump_expr, NULL, "(", "", ", ", ")")
 *  Example: to create the list of declarations of globals, use
 *    UTIL_DUMP_LIST_FIELD(sa, jdf->globals, next, name, dumpstring, NULL, "", "  int ", ";\n", ";\n");
 */
#define UTIL_DUMP_LIST_FIELD(arena, structure_ptr, nextfield, eltfield, fct, fctarg, before, prefix, separator, after) \
    util_dump_list_fct( arena, structure_ptr,                           \
                        (char *)&(structure_ptr->nextfield)-(char *)structure_ptr, \
                        (char *)&(structure_ptr->eltfield)-(char *)structure_ptr, \
                            fct, fctarg, before, prefix, separator, after)

/**
 * UTIL_DUMP_LIST:
 *    Iterate over the elements of a list, transforming each element in a string using a parameter function,
 *    and concatenate all strings.
 *    The function has the prototype  list_elt_t **e, void *a -> char *strelt
 *    The final string has the format
 *       before (prefix strelt separator)* (prefix strelt) after
 *  @param [IN] arena:         string arena to use to add strings elements to the final string
 *  @param [IN] structure_ptr: pointer to a structure that implement any list
 *  @param [IN] nextfield:     the name of a field pointing to the next structure pointer
 *  @param [IN] fct:           a function that transforms a pointer to a list element to a string of characters
 *  @param [IN] fctarg:        fixed argument of the function
 *  @param [IN] before:        string (of characters) representing what must appear before the list
 *  @param [IN] prefix:        string (of characters) representing what must appear before each element
 *  @param [IN] separator:     string (of characters) that will be put between each element, but not at the end
 *                             or before the first
 *  @param [IN] after:         string (of characters) that will be put at the end of the list, after the last
 *                             element
 *
 *  If the function fct return NULL, the element is ignored
 *
 *  @return a string (of characters) written in arena with the list formed so.
 *
 *  Example: to create the list of expressions that is #define list of macros, transforming each element
 *            using both the name of the element and the number of parameters, use
 *          UTIL_DUMP_LIST(sa1, jdf->data, next, dump_data, sa2, "", "#define ", "\n", "\n"));
 */
#define UTIL_DUMP_LIST(arena, structure_ptr, nextfield, fct, fctarg, before, prefix, separator, after) \
    util_dump_list_fct( arena, structure_ptr,                           \
                        (char *)&(structure_ptr->nextfield)-(char *)structure_ptr, \
                        0, \
                        fct, fctarg, before, prefix, separator, after)

/**
 * util_dump_list_fct:
 *   function used by the UTIL_DUMP_LIST* macros. Do not use directly.
 */
static inline char*
util_dump_list_fct(string_arena_t *sa,
                   const void *firstelt, unsigned int next_offset, unsigned int elt_offset,
                   dumper_function_t fct, void *fctarg,
                   const char *before, const char *prefix, const char *separator, const char *after)
{
    char *eltstr;
    const char *prevstr = "";
    void *elt;

    string_arena_init(sa);

    string_arena_add_string(sa, "%s", before);

    while(firstelt != NULL) {
        elt = ((void **)((char*)(firstelt) + elt_offset));
        eltstr = fct(elt, fctarg);

        firstelt = *((void **)((char *)(firstelt) + next_offset));
        if( eltstr != NULL ) {
            string_arena_add_string(sa, "%s%s%s", prevstr, prefix, eltstr);
            prevstr = separator;
        }
    }

    string_arena_add_string(sa, "%s", after);

    return string_arena_get_string(sa);
}

jdf_def_list_t* jdf_create_properties_list( const char* name,
                                            int default_int,
                                            const char* default_char,
                                            jdf_def_list_t* next );

/**
 * Utilities to dump expressions and other parts of the internal
 * storage structure.
 */
typedef struct expr_info {
    struct string_arena* sa;
    const char* prefix;
    char *assignments;
    const char* suffix;
} expr_info_t;
/* The elem should be a jdf_expr_t while the arg should be an expr_info_t */
char * dump_expr(void **elem, void *arg);


#endif
