/*
 * Copyright (c) 2014-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

char *parsec_process_name(void) {
    char name[PATH_MAX];
    char *sname = NULL;
    int ret;
#if defined(__APPLE__)
    uint32_t len = PATH_MAX;
    ret = _NSGetExecutablePath(name, &len);
    if(0 != ret) {
        snprintf(name, len, "parsec-app");
        return strdup(name);
    }
#else
    size_t len = PATH_MAX;
    ret = readlink("/proc/self/exe", name, len);
    if(-1 == ret) {
        snprintf(name, len, "parsec-app" );
        return strdup(name);
    }
    name[ret] = '\0';
#endif
    sname = rindex(name, PARSEC_PATH_SEP[0]);
    if(NULL == sname) return strdup(name);
    else return strdup(sname+1);
}

