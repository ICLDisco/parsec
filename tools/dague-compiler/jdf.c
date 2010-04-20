#include <stdio.h>
#include <stdarg.h>

#include "jdf.h"

jdf_t current_jdf;
int current_lineno;

extern const char *yyfilename;

void jdf_warn(int lineno, const char *format, ...)
{
    char msg[512];
    va_list ap;

    va_start(ap, format);
    vsnprintf(msg, 512, format, ap);
    va_end(ap);

    fprintf(stderr, "Warning on %s:%d: %s\n", yyfilename, lineno, msg);
}

void jdf_prepare_parsing(void)
{
    current_jdf.preambles = NULL;
    current_jdf.globals   = NULL;
    current_jdf.functions = NULL;
    current_lineno = 1;
}

