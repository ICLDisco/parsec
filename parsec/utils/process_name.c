#include "parsec/parsec_config.h"
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

char *parsec_process_name(void) {
    char name[PATH_MAX];
    char *sname = NULL;
    size_t len = PATH_MAX;
    int ret;
#if defined(__APPLE__)
    ret = _NSGetExecutablePath(name, &len);
    if(0 != ret) {
        snprintf(name, len, "parsec-app");
        return strdup(name);
    }
#else
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

