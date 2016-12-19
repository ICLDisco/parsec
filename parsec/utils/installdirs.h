/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef PARSEC_MCA_INSTALLDIRS_INSTALLDIRS_H
#define PARSEC_MCA_INSTALLDIRS_INSTALLDIRS_H

#include <parsec_config.h>

BEGIN_C_DECLS

/**
 * Most of this file is just for ompi_info.  The only public interface
 * once parsec_init has been called is the opal_install_dirs structure
 * and the parsec_install_dirs_expand() call
 */
struct parsec_install_dirs_s {
    char* prefix;
    char* exec_prefix;
    char* bindir;
    char* sbindir;
    char* libexecdir;
    char* datarootdir;
    char* datadir;
    char* sysconfdir;
    char* sharedstatedir;
    char* localstatedir;
    char* libdir;
    char* includedir;
    char* infodir;
    char* mandir;
    char* pkgdatadir;
    char* pkglibdir;
    char* pkgincludedir;
};
typedef struct parsec_install_dirs_s parsec_install_dirs_t;

/**
 * Initialize the installdirs and set the fields to their corresponding values.
 */
PARSEC_DECLSPEC extern int parsec_installdirs_open(void);

/**
 * Release all structures related to the install dirs.
 */
PARSEC_DECLSPEC extern int parsec_installdirs_close(void);

/**
 * Install directories.  Only available after parsec_init()
 */
PARSEC_DECLSPEC extern parsec_install_dirs_t parsec_install_dirs;

END_C_DECLS

#endif /* PARSEC_MCA_INSTALLDIRS_INSTALLDIRS_H */
