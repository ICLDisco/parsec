/**
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _string_arena_h
#define _string_arena_h

#if defined(DAGUE_HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(DAGUE_HAVE_STDARG_H) */

typedef struct string_arena {
    char *ptr;
    int   pos;
    int   size;
} string_arena_t;

static inline string_arena_t *string_arena_new(int base_size)
{
    string_arena_t *sa;
    sa = (string_arena_t*)calloc(1, sizeof(string_arena_t));
    if( base_size == 0 ) {
        base_size = 1;
    }
    sa->ptr  = (char*)malloc(base_size);
    sa->pos  = 0;
    sa->ptr[0]='\0';
    sa->size = base_size;
    return sa;
}

static inline void string_arena_free(string_arena_t *sa)
{
    free(sa->ptr);
    sa->pos  = -1;
    sa->size = -1;
    free(sa);
}

#if defined(__GNUC__)
static inline void string_arena_add_string(string_arena_t *sa, const char *format, ...) __attribute__((format(printf,2,3)));
#endif
static inline void string_arena_add_string(string_arena_t *sa, const char *format, ...)
{
    va_list ap;
    int length;

  redo:
    va_start(ap, format);
    /* we can safely reuse the ap va_list by calling va_end followed by va_start */
    length = vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap);
    if( length >= (sa->size - sa->pos) ) {
        va_end(ap);
        /* realloc */
        sa->size = sa->pos + 4 * length + 1;
        sa->ptr = (char*)realloc( sa->ptr, sa->size );
        goto redo;
    }
    sa->pos += length;

    va_end(ap);
}

static inline void string_arena_init(string_arena_t *sa)
{
    sa->pos = 0;
    sa->ptr[0] = '\0';
}

static inline char *string_arena_get_string(string_arena_t *sa)
{
    return sa->ptr;
}

#endif
