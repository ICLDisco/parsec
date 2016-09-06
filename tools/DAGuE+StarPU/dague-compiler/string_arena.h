#ifndef _string_arena_h
#define _string_arena_h

typedef struct string_arena {
    char *ptr;
    int   pos;
    int   size;
} string_arena_t;

static string_arena_t *string_arena_new(int base_size)
{
    string_arena_t *sa;
    sa = (string_arena_t*)calloc(1, sizeof(string_arena_t));
    if( base_size == 0 ) {
        base_size = 1;
    }
    sa->ptr  = (char*)malloc(base_size);
    sa->pos  = 0;
    sa->size = base_size;
    return sa;
}

static void string_arena_free(string_arena_t *sa)
{
    free(sa->ptr);
    sa->pos  = -1;
    sa->size = -1;
    free(sa);
}

#if defined(__GNUC__)
static void string_arena_add_string(string_arena_t *sa, const char *format, ...) __attribute__((format(printf,2,3)));
#endif
static void string_arena_add_string(string_arena_t *sa, const char *format, ...)
{
    va_list ap, ap2;
    int length;

    va_start(ap, format);
    /* va_list might have pointer to internal state and using
       it twice is a bad idea.  So make a copy for the second
       use.  Copy order taken from Autoconf docs. */
#if defined(DAGUE_HAVE_VA_COPY)
    va_copy(ap2, ap);
#elif defined(DAGUE_HAVE_UNDERSCORE_VA_COPY)
    __va_copy(ap2, ap);
#else
    memcpy (&ap2, &ap, sizeof(va_list));
#endif

    length = vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap);
    if( length >= (sa->size - sa->pos) ) {
        /* realloc */
        sa->size = sa->pos + length + 1;
        sa->ptr = (char*)realloc( sa->ptr, sa->size );
        length = vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap2);
    }
    sa->pos += length;

#if defined(DAGUE_HAVE_VA_COPY) || defined(DAGUE_HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(DAGUE_HAVE_VA_COPY) || defined(DAGUE_HAVE_UNDERSCORE_VA_COPY) */
    va_end(ap);
}

static void string_arena_init(string_arena_t *sa)
{
    sa->pos = 0;
    sa->ptr[0] = '\0';
}

static char *string_arena_get_string(string_arena_t *sa)
{
    return sa->ptr;
}

#endif
