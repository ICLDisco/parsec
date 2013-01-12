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

#ifndef DAGUE_MCA_INSTALLDIRS_INSTALLDIRS_H
#define DAGUE_MCA_INSTALLDIRS_INSTALLDIRS_H

#include <dague_config.h>

BEGIN_C_DECLS

/**
 * Most of this file is just for ompi_info.  The only public interface
 * once dague_init has been called is the opal_install_dirs structure
 * and the dague_install_dirs_expand() call
 */
struct dague_install_dirs_s {
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
typedef struct dague_install_dirs_s dague_install_dirs_t;

/**
 * Install directories.  Only available after dague_init()
 */
DAGUE_DECLSPEC extern dague_install_dirs_t dague_install_dirs;

END_C_DECLS

#endif /* DAGUE_MCA_INSTALLDIRS_INSTALLDIRS_H */
