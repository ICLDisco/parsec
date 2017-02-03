#Doxygen Documentation

Document all functions prototypes, modules, macros, structures and variables. Doxygen is not very flexible: read this documentation and follow the guidelines. Remember to generate the documentation, and check that it appears as expected.

The file `profiling.h` has examples of all features used in the documentation.

## Functions Prototype Documentation
To document a function, use the following format:

```
/**
 * @brief Single line explanation of the function
 * 
 * @details
 * Multiple lines explaining the function behavior
 * exceptions, preconditions, etc.
 * 
 * @param[in]    name_of_parameter explanation of parameter
 * @param[inout] name_of_parameter explanation of parameter
 * @return return value, if there is one
 */
int function(int param1, int *param2);
```

`[in]` **must** be attached to `@param`, and **it must** be *lower case*. Otherwise, doxygen thinks the name of the parameter is `[IN]`.

Blank lines are *important*: there must be a blank line between `@brief` and `@details`.

## Structures documentation

Use the following format:

```
/**
 * General description of the structure
 */
struct name_s {
    type1  field1;  /**< What is stored in this field */
    type2  field2;  /**< What is stored in this field */
};
```

It is important to use the `/**<` marker, or the description of the field is not imported in an array representation of the data structure.

## Macros Documentation

Use the following format:

```
/** @brief document the macro */
#define macro 
```

Without the `@brief`, the documentation does not appear just under the macro definition. All macro definitions are output directly in the generated documentation.

## Variable Documentation

Use the following format:

```
/** @brief document the variable */
int var;
```

Without the `@brief`, the documentation does not appear just under the variable definition.

## Modules Documentation

General documentation on how the functions of a same module are used together belong to the Module documentation section. Modules are defined with the following format:

```
/**
 * @defgroup name_of_module Readable Name Of Module
 * @ingroup name_of_parent_module
 * @{
 * 
 * @brief One line description of module
 * 
 * @details multiple lines description of module
 * Can use markdown language features, like 
 * sharp for section headers, etc.
 */
 ...
 /** @} */
```

All functions between `@{` and `@}` belong to the same Module. Most modules have already been created, look for defgroup. Some of them are defined in the file `groups.dox`, because they span over multiple files in the source code.

## DOs and DO NOTs

### Functions Prototypes
**DO** comment all function prototypes at their prototype definition (typically .h file), or, if they are static functions used in a single file, before the function definition. **DO NOT** comment *both* the function definition and the function prototype, or the documentation of the function will be duplicated.

### Doxygen Comments
All `/** ... */` comments are interpreted by doxygen as to be added to the documentation.

If you enter a `/** ... */` comment without doxygen commands in it, all the text of the comment will be appended to the next doxygen group.

So, **DO NOT** start comments inside the code with `/**`, despite the nice coloring of your editor. All these comments make no sense once they are appended to the next function or structure documentation.

**DO** configure your editor to use your preferred coloring scheme for normal comments :)

### Doxygen keywords
Any `@something` inside a `/** ... */` block is interpreted as a Doxygen keyword. Doxygen stops parsing the entire file if it finds a keyword it does not know. **DO NOT** use your own keywords, **DO** check the generated documentation.

## Modules Structure
All functions, macros, variables should belong to a module. **DO** look at the documentation to see the structure, and if you document a function, be sure that it belongs to the right module. **DO NOT** just document a function without enclosing it inside a `@ingroup @{` ... `@}` group, or without adding a `@addtogroup group_name` before the function definition.
