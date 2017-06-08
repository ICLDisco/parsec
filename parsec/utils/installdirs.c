/*
 * Copyright (c) 2012-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <stdio.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#if defined(PARSEC_HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(PARSEC_HAVE_STDBOOL_H) */

#include "parsec/constants.h"
#include "parsec/utils/installdirs.h"
#include "parsec/utils/os_path.h"

parsec_install_dirs_t parsec_install_dirs = {
    NULL, /* prefix */
    NULL, /* exec_prefix */
    NULL, /* bindir */
    NULL, /* sbindir */
    NULL, /* libexecdir */
    NULL, /* datarootdir */
    NULL, /* datadir */
    NULL, /* sysconfdir */
    NULL, /* sharedstatedir */
    NULL, /* localstatedir */
    NULL, /* libdir */
    NULL, /* includedir */
    NULL, /* infodir */
    NULL, /* mandir */
    NULL, /* pkgdatadir */
    NULL, /* pkglibdir */
    NULL, /* pkgincludedir */
};

#define SET_FIELD(root, field, relative_dir)                              \
    do {                                                                  \
        char* _root = (root);                                             \
        if (NULL != (_root) && 0 != strlen(_root)) {                      \
            char *path =                                                  \
                (char *)malloc(strlen(_root) + strlen(relative_dir) + 1); \
            strcpy(path, _root);                                          \
            strcat(path, relative_dir);                                   \
            if(NULL != parsec_install_dirs.field)                          \
                free(parsec_install_dirs.field);                           \
            parsec_install_dirs.field = path;                              \
        }                                                                 \
    } while (0)

#define INSTALLDIRS_EXPAND_INTERNAL(field)                                \
    do {                                                                  \
        char* retval = parsec_installdirs_expand_internal(parsec_install_dirs.field, true); \
        if(NULL != retval) {                                              \
            if(NULL != parsec_install_dirs.field)                          \
                free(parsec_install_dirs.field);                           \
            parsec_install_dirs.field = retval;                            \
        }                                                                 \
    } while (0)


#ifdef __WINDOWS__

static int
parsec_installdirs_windows(void)
{
    /* check the env first */
    char* parsec_home = getenv("PARSEC_HOME");

    /* if OPENMPI_HOME is not set, check the registry */
    if(NULL == parsec_home) {
        HKEY parsec_key;
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
        if( ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, "Software\\PARSEC", 0, KEY_READ, &parsec_key) ||
            ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, "Software\\Wow6432Node\\PARSEC", 0, KEY_READ, &parsec_key) ) {
                for( i = 0; true; i++) {
                valueLength = 1024;
                valueName[0] = '\0';
                cbData = 1024;
                valueLength = 1024;
                if( ERROR_SUCCESS == RegEnumValue( (parsec_key), i, valueName, &valueLength,
                                                    NULL, &keyType, (LPBYTE) vData, &cbData ) ) {
                    if( ((REG_EXPAND_SZ == keyType) || (REG_SZ == keyType)) &&
                        (0 == strncasecmp( valueName, ("PARSEC_HOME"), strlen(("PARSEC_HOME")) )) ) {
                        parsec_home = strdup(vData);
                        break;
                    }
                } else
                    break;
            }
        }

        RegCloseKey(parsec_key);
    }

    SET_FIELD(parsec_home, prefix, "");
    SET_FIELD(parsec_home, exec_prefix, "/bin");
    SET_FIELD(parsec_home, bindir, "/bin");
    SET_FIELD(parsec_home, sbindir, "/sbin");
    SET_FIELD(parsec_home, libexecdir, "/libexec");
    SET_FIELD(parsec_home, datarootdir, "/share");
    SET_FIELD(parsec_home, datadir, "/share");
    SET_FIELD(parsec_home, sysconfdir, "/etc");
    SET_FIELD(parsec_home, sharedstatedir, "/com");
    SET_FIELD(parsec_home, localstatedir, "/var");
    SET_FIELD(parsec_home, libdir, "/lib");
    SET_FIELD(parsec_home, includedir, "/include");
    SET_FIELD(parsec_home, infodir, "/share/info");
    SET_FIELD(parsec_home, mandir, "/share/man");
    SET_FIELD(parsec_home, pkgdatadir, "/share/parsec");
    SET_FIELD(parsec_home, pkglibdir, "/lib/parsec");
    SET_FIELD(parsec_home, pkgincludedir, "/include/parsec");

    return PARSEC_SUCCESS;
}

#endif  /* !defined(__WINDOWS__) */

static int parsec_installdirs_from_env(void)
{
    SET_FIELD(getenv("PARSEC_PREFIX"), prefix, "");
    SET_FIELD(getenv("PARSEC_EXEC_PREFIX"), exec_prefix, "");
    SET_FIELD(getenv("PARSEC_BINDIR"), bindir, "");
    SET_FIELD(getenv("PARSEC_SBINDIR"), sbindir, "");
    SET_FIELD(getenv("PARSEC_LIBEXECDIR"), libexecdir, "");
    SET_FIELD(getenv("PARSEC_DATAROOTDIR"), datarootdir, "");
    SET_FIELD(getenv("PARSEC_DATADIR"), datadir, "");
    SET_FIELD(getenv("PARSEC_SYSCONFDIR"), sysconfdir, "");
    SET_FIELD(getenv("PARSEC_SHAREDSTATEDIR"), sharedstatedir, "");
    SET_FIELD(getenv("PARSEC_LOCALSTATEDIR"), localstatedir, "");
    SET_FIELD(getenv("PARSEC_LIBDIR"), libdir, "");
    SET_FIELD(getenv("PARSEC_INCLUDEDIR"), includedir, "");
    SET_FIELD(getenv("PARSEC_INFODIR"), infodir, "");
    SET_FIELD(getenv("PARSEC_MANDIR"), mandir, "");
    SET_FIELD(getenv("PARSEC_PKGDATADIR"), pkgdatadir, "");
    SET_FIELD(getenv("PARSEC_PKGLIBDIR"), pkglibdir, "");
    SET_FIELD(getenv("PARSEC_PKGINCLUDEDIR"), pkgincludedir, "");

    return PARSEC_SUCCESS;
}

#define EXPAND_STRING(field)                                            \
    do {                                                                \
        if (NULL != (start_pos = strstr(retval, "${" #field "}"))) {    \
            int rc;                                                     \
            tmp = retval;                                               \
            *start_pos = '\0';                                          \
            end_pos = start_pos + strlen("${" #field "}");              \
            rc = asprintf(&retval, "%s%s%s", tmp,                       \
                     parsec_install_dirs.field + destdir_offset,         \
                     end_pos);                                          \
            free(tmp);                                                  \
            changed = true;                                             \
            (void)rc;                                                   \
        }                                                               \
    } while (0)


/*
 * Read the lengthy comment below to understand the value of the
 * is_setup parameter.
 */
static char *
parsec_installdirs_expand_internal(const char* input, bool is_setup)
{
    size_t len, i;
    bool needs_expand = false;
    char *retval = NULL;
    char *destdir = NULL;
    size_t destdir_offset = 0;

    /* This is subtle, and worth explaining.

       If we substitute in any ${FIELD} values, we need to prepend it
       with the value of the $PARSEC_DESTDIR environment variable -- if
       it is set.

       We need to handle at least three cases properly (assume that
       configure was invoked with --prefix=/opt/parsec and no other
       directory specifications, and PARSEC_DESTDIR is set to
       /tmp/buildroot):

       1. Individual directories, such as libdir.  These need to be
          prepended with DESTDIR.  I.e., return
          /tmp/buildroot/opt/parsec/lib.

       2. Compiler flags that have ${FIELD} values embedded in them.
          For example, consider if a wrapper compiler data file
          contains the line:

          preprocessor_flags=-DMYFLAG="${prefix}/share/randomthingy/"

          The value we should return is:

          -DMYFLAG="/tmp/buildroot/opt/parsec/share/randomthingy/"

       3. Compiler flags that do not have any ${FIELD} values.
          For example, consider if a wrapper compiler data file
          contains the line:

          preprocessor_flags=-pthread

          The value we should return is:

          -pthread

       Note, too, that this PARSEC_DESTDIR futzing only needs to occur
       during opal_init().  By the time opal_init() has completed, all
       values should be substituted in that need substituting.  Hence,
       we take an extra parameter (is_setup) to know whether we should
       do this futzing or not. */
    if (is_setup) {
        destdir = getenv("PARSEC_DESTDIR");
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
        retval = parsec_os_path(false, destdir, tmp, NULL);
        free(tmp);
    }

    return retval;
}

int
parsec_installdirs_open(void)
{
    /* Step one, get everything from the default configuration */
#ifdef __WINDOWS__
    /* On Windows the default installation is specified in the registry */
    parsec_installdirs_windows()
#else
    /* Otherwise it can be deducted at compile time */
    SET_FIELD(PARSEC_INSTALL_PREFIX, prefix, "");
    SET_FIELD(PARSEC_INSTALL_PREFIX, exec_prefix, "/bin");
    SET_FIELD(PARSEC_INSTALL_PREFIX, bindir, "/bin");
    SET_FIELD(PARSEC_INSTALL_PREFIX, sbindir, "/sbin");
    SET_FIELD(PARSEC_INSTALL_PREFIX, libexecdir, "/libexec");
    SET_FIELD(PARSEC_INSTALL_PREFIX, datarootdir, "/share");
    SET_FIELD(PARSEC_INSTALL_PREFIX, datadir, "/share");
    SET_FIELD(PARSEC_INSTALL_PREFIX, sysconfdir, "/etc");
    SET_FIELD(PARSEC_INSTALL_PREFIX, sharedstatedir, "/com");
    SET_FIELD(PARSEC_INSTALL_PREFIX, localstatedir, "/var");
    SET_FIELD(PARSEC_INSTALL_PREFIX, libdir, "/lib");
    SET_FIELD(PARSEC_INSTALL_PREFIX, includedir, "/include");
    SET_FIELD(PARSEC_INSTALL_PREFIX, infodir, "/share/info");
    SET_FIELD(PARSEC_INSTALL_PREFIX, mandir, "/share/man");
    SET_FIELD(PARSEC_INSTALL_PREFIX, pkgdatadir, "/share/parsec");
    SET_FIELD(PARSEC_INSTALL_PREFIX, pkglibdir, "/lib/parsec");
    SET_FIELD(PARSEC_INSTALL_PREFIX, pkgincludedir, "/include/parsec");
#endif  /* defined(__WINDOS__) */

    /* Now get anyting specified by the environment */
    parsec_installdirs_from_env();

    /* expand out all the fields */
    INSTALLDIRS_EXPAND_INTERNAL(prefix);
    INSTALLDIRS_EXPAND_INTERNAL(exec_prefix);
    INSTALLDIRS_EXPAND_INTERNAL(bindir);
    INSTALLDIRS_EXPAND_INTERNAL(sbindir);
    INSTALLDIRS_EXPAND_INTERNAL(libexecdir);
    INSTALLDIRS_EXPAND_INTERNAL(datarootdir);
    INSTALLDIRS_EXPAND_INTERNAL(datadir);
    INSTALLDIRS_EXPAND_INTERNAL(sysconfdir);
    INSTALLDIRS_EXPAND_INTERNAL(sharedstatedir);
    INSTALLDIRS_EXPAND_INTERNAL(localstatedir);
    INSTALLDIRS_EXPAND_INTERNAL(libdir);
    INSTALLDIRS_EXPAND_INTERNAL(includedir);
    INSTALLDIRS_EXPAND_INTERNAL(infodir);
    INSTALLDIRS_EXPAND_INTERNAL(mandir);
    INSTALLDIRS_EXPAND_INTERNAL(pkgdatadir);
    INSTALLDIRS_EXPAND_INTERNAL(pkglibdir);
    INSTALLDIRS_EXPAND_INTERNAL(pkgincludedir);

#if 0
    fprintf(stderr, "prefix:         %s\n", parsec_install_dirs.prefix);
    fprintf(stderr, "exec_prefix:    %s\n", parsec_install_dirs.exec_prefix);
    fprintf(stderr, "bindir:         %s\n", parsec_install_dirs.bindir);
    fprintf(stderr, "sbindir:        %s\n", parsec_install_dirs.sbindir);
    fprintf(stderr, "libexecdir:     %s\n", parsec_install_dirs.libexecdir);
    fprintf(stderr, "datarootdir:    %s\n", parsec_install_dirs.datarootdir);
    fprintf(stderr, "datadir:        %s\n", parsec_install_dirs.datadir);
    fprintf(stderr, "sysconfdir:     %s\n", parsec_install_dirs.sysconfdir);
    fprintf(stderr, "sharedstatedir: %s\n", parsec_install_dirs.sharedstatedir);
    fprintf(stderr, "localstatedir:  %s\n", parsec_install_dirs.localstatedir);
    fprintf(stderr, "libdir:         %s\n", parsec_install_dirs.libdir);
    fprintf(stderr, "includedir:     %s\n", parsec_install_dirs.includedir);
    fprintf(stderr, "infodir:        %s\n", parsec_install_dirs.infodir);
    fprintf(stderr, "mandir:         %s\n", parsec_install_dirs.mandir);
    fprintf(stderr, "pkgdatadir:     %s\n", parsec_install_dirs.pkgdatadir);
    fprintf(stderr, "pkglibdir:      %s\n", parsec_install_dirs.pkglibdir);
    fprintf(stderr, "pkgincludedir:  %s\n", parsec_install_dirs.pkgincludedir);
#endif

    return PARSEC_SUCCESS;
}

int parsec_installdirs_close(void)
{
    free(parsec_install_dirs.prefix);         parsec_install_dirs.prefix = NULL;
    free(parsec_install_dirs.exec_prefix);    parsec_install_dirs.exec_prefix = NULL;
    free(parsec_install_dirs.bindir);         parsec_install_dirs.bindir = NULL;
    free(parsec_install_dirs.sbindir);        parsec_install_dirs.sbindir = NULL;
    free(parsec_install_dirs.libexecdir);     parsec_install_dirs.libexecdir = NULL;
    free(parsec_install_dirs.datarootdir);    parsec_install_dirs.datarootdir = NULL;
    free(parsec_install_dirs.datadir);        parsec_install_dirs.datadir = NULL;
    free(parsec_install_dirs.sysconfdir);     parsec_install_dirs.sysconfdir = NULL;
    free(parsec_install_dirs.sharedstatedir); parsec_install_dirs.sharedstatedir = NULL;
    free(parsec_install_dirs.localstatedir);  parsec_install_dirs.localstatedir = NULL;
    free(parsec_install_dirs.libdir);         parsec_install_dirs.libdir = NULL;
    free(parsec_install_dirs.includedir);     parsec_install_dirs.includedir = NULL;
    free(parsec_install_dirs.infodir);        parsec_install_dirs.infodir = NULL;
    free(parsec_install_dirs.mandir);         parsec_install_dirs.mandir = NULL;
    free(parsec_install_dirs.pkgdatadir);     parsec_install_dirs.pkgdatadir = NULL;
    free(parsec_install_dirs.pkglibdir);      parsec_install_dirs.pkglibdir = NULL;
    free(parsec_install_dirs.pkgincludedir);  parsec_install_dirs.pkgincludedir = NULL;

    return PARSEC_SUCCESS;
}
