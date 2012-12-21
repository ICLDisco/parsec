/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 */

#include "dague_config.h"

#include <stdlib.h>
#include <stdio.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#if defined(HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(HAVE_STDBOOL_H) */

#include "dague/constants.h"
#include "dague/utils/installdirs.h"
#include "dague/utils/os_path.h"

dague_install_dirs_t dague_install_dirs;

#define SET_FIELD(root, field, relative_dir)                              \
    do {                                                                  \
        char* _root = (root);                                             \
        if (NULL != (_root) && 0 != strlen(_root)) {                      \
            char *path =                                                  \
                (char *)malloc(strlen(_root) + strlen(relative_dir) + 1); \
            strcpy(path, _root);                                          \
            strcat(path, relative_dir);                                   \
            if(NULL != dague_install_dirs.field)                          \
                free(dague_install_dirs.field);                           \
            dague_install_dirs.field = path;                              \
        }                                                                 \
    } while (0)

#ifdef __WINDOWS__

int
dague_installdirs_windows(void)
{
    /* check the env first */
    char* dague_home = getenv("DAGUE_HOME");

    /* if OPENMPI_HOME is not set, check the registry */
    if(NULL == dague_home) {
        HKEY dague_key;
        int i;
        DWORD cbData, valueLength, keyType;
        char valueName[1024], vData[1024];

        /* The OPENMPI_HOME is the only one which is required to be in the registry.
         * All others can be composed starting from OPAL_PREFIX.
         *
         * On 32 bit Windows, we write in HKEY_LOCAL_MACHINE\Software\Open MPI,
         * but on 64 bit Windows, we always use HKEY_LOCAL_MACHINE\Software\Wow6432Node\Open MPI
         * for both 32 and 64 bit OMPI, because we only have 32 bit installer, and Windows will
         * always consider OMPI as 32 bit application.
         */
        if( ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, "Software\\DAGUE", 0, KEY_READ, &dague_key) ||
            ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, "Software\\Wow6432Node\\DAGUE", 0, KEY_READ, &dague_key) ) {    
                for( i = 0; true; i++) {
                valueLength = 1024;
                valueName[0] = '\0';
                cbData = 1024;
                valueLength = 1024;
                if( ERROR_SUCCESS == RegEnumValue( (dague_key), i, valueName, &valueLength,
                                                    NULL, &keyType, (LPBYTE) vData, &cbData ) ) {
                    if( ((REG_EXPAND_SZ == keyType) || (REG_SZ == keyType)) &&
                        (0 == strncasecmp( valueName, ("DAGUE_HOME"), strlen(("DAGUE_HOME")) )) ) {
                        dague_home = strdup(vData);
                        break;
                    }
                } else
                    break;
            }
        }

        RegCloseKey(dague_key);
    }

    SET_FIELD(dague_home, prefix, "");
    SET_FIELD(dague_home, exec_prefix, "/bin");
    SET_FIELD(dague_home, bindir, "/bin");
    SET_FIELD(dague_home, sbindir, "/sbin");
    SET_FIELD(dague_home, libexecdir, "/libexec");
    SET_FIELD(dague_home, datarootdir, "/share");
    SET_FIELD(dague_home, datadir, "/share");
    SET_FIELD(dague_home, sysconfdir, "/etc");
    SET_FIELD(dague_home, sharedstatedir, "/com");
    SET_FIELD(dague_home, localstatedir, "/var");
    SET_FIELD(dague_home, libdir, "/lib");
    SET_FIELD(dague_home, includedir, "/include");
    SET_FIELD(dague_home, infodir, "/share/info");
    SET_FIELD(dague_home, mandir, "/share/man");
    SET_FIELD(dague_home, pkgdatadir, "/share/dague");
    SET_FIELD(dague_home, pkglibdir, "/lib/dague");
    SET_FIELD(dague_home, pkgincludedir, "/include/dague");

    return DAGUE_SUCCESS;
}

#endif  /* !defined(__WINDOWS__) */

static int dague_installdirs_from_env(void)
{
    SET_FIELD(getenv("DAGUE_PREFIX"), prefix, "");
    SET_FIELD(getenv("DAGUE_EXEC_PREFIX"), exec_prefix, "");
    SET_FIELD(getenv("DAGUE_BINDIR"), bindir, "");
    SET_FIELD(getenv("DAGUE_SBINDIR"), sbindir, "");
    SET_FIELD(getenv("DAGUE_LIBEXECDIR"), libexecdir, "");
    SET_FIELD(getenv("DAGUE_DATAROOTDIR"), datarootdir, "");
    SET_FIELD(getenv("DAGUE_DATADIR"), datadir, "");
    SET_FIELD(getenv("DAGUE_SYSCONFDIR"), sysconfdir, "");
    SET_FIELD(getenv("DAGUE_SHAREDSTATEDIR"), sharedstatedir, "");
    SET_FIELD(getenv("DAGUE_LOCALSTATEDIR"), localstatedir, "");
    SET_FIELD(getenv("DAGUE_LIBDIR"), libdir, "");
    SET_FIELD(getenv("DAGUE_INCLUDEDIR"), includedir, "");
    SET_FIELD(getenv("DAGUE_INFODIR"), infodir, "");
    SET_FIELD(getenv("DAGUE_MANDIR"), mandir, "");
    SET_FIELD(getenv("DAGUE_PKGDATADIR"), pkgdatadir, "");
    SET_FIELD(getenv("DAGUE_PKGLIBDIR"), pkglibdir, "");
    SET_FIELD(getenv("DAGUE_PKGINCLUDEDIR"), pkgincludedir, "");

    return DAGUE_SUCCESS;
}

#define EXPAND_STRING(field)                                            \
    do {                                                                \
        if (NULL != (start_pos = strstr(retval, "${" #field "}"))) {    \
            tmp = retval;                                               \
            *start_pos = '\0';                                          \
            end_pos = start_pos + strlen("${" #field "}");              \
            asprintf(&retval, "%s%s%s", tmp,                            \
                     dague_install_dirs.field + destdir_offset,         \
                     end_pos);                                          \
            free(tmp);                                                  \
            changed = true;                                             \
        }                                                               \
    } while (0)


/*
 * Read the lengthy comment below to understand the value of the
 * is_setup parameter.
 */
static char *
dague_installdirs_expand_internal(const char* input, bool is_setup)
{
    size_t len, i;
    bool needs_expand = false;
    char *retval = NULL;
    char *destdir = NULL;
    size_t destdir_offset = 0;

    /* This is subtle, and worth explaining.  

       If we substitute in any ${FIELD} values, we need to prepend it
       with the value of the $DAGUE_DESTDIR environment variable -- if
       it is set.  

       We need to handle at least three cases properly (assume that
       configure was invoked with --prefix=/opt/dague and no other
       directory specifications, and DAGUE_DESTDIR is set to
       /tmp/buildroot):

       1. Individual directories, such as libdir.  These need to be
          prepended with DESTDIR.  I.e., return
          /tmp/buildroot/opt/dague/lib.

       2. Compiler flags that have ${FIELD} values embedded in them.
          For example, consider if a wrapper compiler data file
          contains the line:

          preprocessor_flags=-DMYFLAG="${prefix}/share/randomthingy/"

          The value we should return is:

          -DMYFLAG="/tmp/buildroot/opt/dague/share/randomthingy/"

       3. Compiler flags that do not have any ${FIELD} values.
          For example, consider if a wrapper compiler data file
          contains the line:

          preprocessor_flags=-pthread

          The value we should return is:

          -pthread

       Note, too, that this DAGUE_DESTDIR futzing only needs to occur
       during opal_init().  By the time opal_init() has completed, all
       values should be substituted in that need substituting.  Hence,
       we take an extra parameter (is_setup) to know whether we should
       do this futzing or not. */
    if (is_setup) {
        destdir = getenv("DAGUE_DESTDIR");
        if (NULL != destdir && strlen(destdir) > 0) {
            destdir_offset = strlen(destdir);
        }
    }

    len = strlen(input);
    for (i = 0 ; i < len ; ++i) {
        if (input[i] == '$') {
            needs_expand = true;
            break;
        }
    }

    retval = strdup(input);
    if (NULL == retval) return NULL;

    if (needs_expand) {
        bool changed = false;
        char *start_pos, *end_pos, *tmp;

        do {
            changed = false;
            EXPAND_STRING(prefix);
            EXPAND_STRING(exec_prefix);
            EXPAND_STRING(bindir);
            EXPAND_STRING(sbindir);
            EXPAND_STRING(libexecdir);
            EXPAND_STRING(datarootdir);
            EXPAND_STRING(datadir);
            EXPAND_STRING(sysconfdir);
            EXPAND_STRING(sharedstatedir);
            EXPAND_STRING(localstatedir);
            EXPAND_STRING(libdir);
            EXPAND_STRING(includedir);
            EXPAND_STRING(infodir);
            EXPAND_STRING(mandir);
            EXPAND_STRING(pkgdatadir);
            EXPAND_STRING(pkglibdir);
            EXPAND_STRING(pkgincludedir);
        } while (changed);
    }

    if (NULL != destdir) {
        char *tmp = retval;
        retval = dague_os_path(false, destdir, tmp, NULL);
        free(tmp);
    }

    return retval;
}

#define CONDITIONAL_COPY(target, origin, field)                 \
    do {                                                        \
        if (origin.field != NULL && target.field == NULL) {     \
            target.field = origin.field;                        \
        }                                                       \
    } while (0)

int
dague_installdirs_open(void)
{
    /* Step one, get everything from the default configuration */
#ifdef __WINDOWS__
    /* On Windows the default installation is specified in the registry */
    dague_installdirs_windows()
#else
    /* Otherwise it can be deducted at compile time */
    SET_FIELD(DAGUE_INSTALL_PREFIX, prefix, "");
    SET_FIELD(DAGUE_INSTALL_PREFIX, exec_prefix, "/bin");
    SET_FIELD(DAGUE_INSTALL_PREFIX, bindir, "/bin");
    SET_FIELD(DAGUE_INSTALL_PREFIX, sbindir, "/sbin");
    SET_FIELD(DAGUE_INSTALL_PREFIX, libexecdir, "/libexec");
    SET_FIELD(DAGUE_INSTALL_PREFIX, datarootdir, "/share");
    SET_FIELD(DAGUE_INSTALL_PREFIX, datadir, "/share");
    SET_FIELD(DAGUE_INSTALL_PREFIX, sysconfdir, "/etc");
    SET_FIELD(DAGUE_INSTALL_PREFIX, sharedstatedir, "/com");
    SET_FIELD(DAGUE_INSTALL_PREFIX, localstatedir, "/var");
    SET_FIELD(DAGUE_INSTALL_PREFIX, libdir, "/lib");
    SET_FIELD(DAGUE_INSTALL_PREFIX, includedir, "/include");
    SET_FIELD(DAGUE_INSTALL_PREFIX, infodir, "/share/info");
    SET_FIELD(DAGUE_INSTALL_PREFIX, mandir, "/share/man");
    SET_FIELD(DAGUE_INSTALL_PREFIX, pkgdatadir, "/share/dague");
    SET_FIELD(DAGUE_INSTALL_PREFIX, pkglibdir, "/lib/dague");
    SET_FIELD(DAGUE_INSTALL_PREFIX, pkgincludedir, "/include/dague");
#endif  /* defined(__WINDOS__) */

    /* Now get anyting specified by the environment */
    dague_installdirs_from_env();

    /* expand out all the fields */
    dague_install_dirs.prefix = 
        dague_installdirs_expand_internal(dague_install_dirs.prefix, true);
    dague_install_dirs.exec_prefix = 
        dague_installdirs_expand_internal(dague_install_dirs.exec_prefix, true);
    dague_install_dirs.bindir = 
        dague_installdirs_expand_internal(dague_install_dirs.bindir, true);
    dague_install_dirs.sbindir = 
        dague_installdirs_expand_internal(dague_install_dirs.sbindir, true);
    dague_install_dirs.libexecdir = 
        dague_installdirs_expand_internal(dague_install_dirs.libexecdir, true);
    dague_install_dirs.datarootdir = 
        dague_installdirs_expand_internal(dague_install_dirs.datarootdir, true);
    dague_install_dirs.datadir = 
        dague_installdirs_expand_internal(dague_install_dirs.datadir, true);
    dague_install_dirs.sysconfdir = 
        dague_installdirs_expand_internal(dague_install_dirs.sysconfdir, true);
    dague_install_dirs.sharedstatedir = 
        dague_installdirs_expand_internal(dague_install_dirs.sharedstatedir, true);
    dague_install_dirs.localstatedir = 
        dague_installdirs_expand_internal(dague_install_dirs.localstatedir, true);
    dague_install_dirs.libdir = 
        dague_installdirs_expand_internal(dague_install_dirs.libdir, true);
    dague_install_dirs.includedir = 
        dague_installdirs_expand_internal(dague_install_dirs.includedir, true);
    dague_install_dirs.infodir = 
        dague_installdirs_expand_internal(dague_install_dirs.infodir, true);
    dague_install_dirs.mandir = 
        dague_installdirs_expand_internal(dague_install_dirs.mandir, true);
    dague_install_dirs.pkgdatadir = 
        dague_installdirs_expand_internal(dague_install_dirs.pkgdatadir, true);
    dague_install_dirs.pkglibdir = 
        dague_installdirs_expand_internal(dague_install_dirs.pkglibdir, true);
    dague_install_dirs.pkgincludedir = 
        dague_installdirs_expand_internal(dague_install_dirs.pkgincludedir, true);

#if 0
    fprintf(stderr, "prefix:         %s\n", dague_install_dirs.prefix);
    fprintf(stderr, "exec_prefix:    %s\n", dague_install_dirs.exec_prefix);
    fprintf(stderr, "bindir:         %s\n", dague_install_dirs.bindir);
    fprintf(stderr, "sbindir:        %s\n", dague_install_dirs.sbindir);
    fprintf(stderr, "libexecdir:     %s\n", dague_install_dirs.libexecdir);
    fprintf(stderr, "datarootdir:    %s\n", dague_install_dirs.datarootdir);
    fprintf(stderr, "datadir:        %s\n", dague_install_dirs.datadir);
    fprintf(stderr, "sysconfdir:     %s\n", dague_install_dirs.sysconfdir);
    fprintf(stderr, "sharedstatedir: %s\n", dague_install_dirs.sharedstatedir);
    fprintf(stderr, "localstatedir:  %s\n", dague_install_dirs.localstatedir);
    fprintf(stderr, "libdir:         %s\n", dague_install_dirs.libdir);
    fprintf(stderr, "includedir:     %s\n", dague_install_dirs.includedir);
    fprintf(stderr, "infodir:        %s\n", dague_install_dirs.infodir);
    fprintf(stderr, "mandir:         %s\n", dague_install_dirs.mandir);
    fprintf(stderr, "pkgdatadir:     %s\n", dague_install_dirs.pkgdatadir);
    fprintf(stderr, "pkglibdir:      %s\n", dague_install_dirs.pkglibdir);
    fprintf(stderr, "pkgincludedir:  %s\n", dague_install_dirs.pkgincludedir);
#endif

    return DAGUE_SUCCESS;
}

int dague_installdirs_close(void)
{
    free(dague_install_dirs.prefix);         dague_install_dirs.prefix = NULL;
    free(dague_install_dirs.exec_prefix);    dague_install_dirs.exec_prefix = NULL;
    free(dague_install_dirs.bindir);         dague_install_dirs.bindir = NULL;
    free(dague_install_dirs.sbindir);        dague_install_dirs.sbindir = NULL;
    free(dague_install_dirs.libexecdir);     dague_install_dirs.libexecdir = NULL;
    free(dague_install_dirs.datarootdir);    dague_install_dirs.datarootdir = NULL;
    free(dague_install_dirs.datadir);        dague_install_dirs.datadir = NULL;
    free(dague_install_dirs.sysconfdir);     dague_install_dirs.sysconfdir = NULL;
    free(dague_install_dirs.sharedstatedir); dague_install_dirs.sharedstatedir = NULL;
    free(dague_install_dirs.localstatedir);  dague_install_dirs.localstatedir = NULL;
    free(dague_install_dirs.libdir);         dague_install_dirs.libdir = NULL;
    free(dague_install_dirs.includedir);     dague_install_dirs.includedir = NULL;
    free(dague_install_dirs.infodir);        dague_install_dirs.infodir = NULL;
    free(dague_install_dirs.mandir);         dague_install_dirs.mandir = NULL;
    free(dague_install_dirs.pkgdatadir);     dague_install_dirs.pkgdatadir = NULL;
    free(dague_install_dirs.pkglibdir);      dague_install_dirs.pkglibdir = NULL;
    free(dague_install_dirs.pkgincludedir);  dague_install_dirs.pkgincludedir = NULL;

    return DAGUE_SUCCESS;
}

