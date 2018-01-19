/*
 * Copyright (c) 2004-2010 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2008 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "parsec/parsec_config.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#ifdef PARSEC_HAVE_SYSLOG_H
#include <syslog.h>
#endif
#include <string.h>
#include <fcntl.h>
#ifdef PARSEC_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef PARSEC_HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#include "parsec/utils/parsec_environ.h"
#include "parsec/utils/output.h"
#include "parsec/constants.h"

/*
 * Private data
 */
static int verbose_stream = -1;
static parsec_output_stream_t verbose;
static char *output_dir = NULL;
static char *output_prefix = NULL;

/*
 * Internal data structures and helpers for the generalized output
 * stream mechanism.
 */
typedef struct {
    bool ldi_used;
    bool ldi_enabled;
    int ldi_verbose_level;

    bool ldi_syslog;
    int ldi_syslog_priority;

#ifndef __WINDOWS__
    char *ldi_syslog_ident;
#else
    HANDLE ldi_syslog_ident;
#endif
    char *ldi_prefix;
    int ldi_prefix_len;

    char *ldi_suffix;
    int ldi_suffix_len;

    bool ldi_stdout;
    bool ldi_stderr;

    bool ldi_file;
    bool ldi_file_want_append;
    char *ldi_file_suffix;
    int ldi_fd;
    int ldi_file_num_lines_lost;
} output_desc_t;

/*
 * Private functions
 */
static void construct(parsec_object_t *stream);
static int do_open(int output_id, parsec_output_stream_t * lds);
static int open_file(int i);
static void free_descriptor(int output_id);
static int make_string(char **no_newline_string, output_desc_t *ldi,
                       const char *format, va_list arglist);
static int output(int output_id, const char *format, va_list arglist);


#define PARSEC_OUTPUT_MAX_STREAMS 64
#if defined(__WINDOWS__) || defined(PARSEC_HAVE_SYSLOG)
#define USE_SYSLOG 1
#else
#define USE_SYSLOG 0
#endif

/* global state */
bool parsec_output_redirected_to_syslog = false;
int parsec_output_redirected_syslog_pri;

/*
 * Local state
 */
static bool initialized = false;
static int default_stderr_fd = -1;
static output_desc_t info[PARSEC_OUTPUT_MAX_STREAMS];
static char *temp_str = 0;
static size_t temp_str_len = 0;
static parsec_atomic_lock_t mutex = { PARSEC_ATOMIC_UNLOCKED };
#if defined(PARSEC_HAVE_SYSLOG)
static bool syslog_opened = false;
#endif  /* defined(PARSEC_HAVE_SYSLOG) */
static char *redirect_syslog_ident = NULL;

OBJ_CLASS_INSTANCE(parsec_output_stream_t, parsec_object_t, construct, NULL);

/*
 * Setup the output stream infrastructure
 */
bool parsec_output_init(void)
{
    int i, rc;
    char hostname[32];
    char *str;

    if (initialized) {
        return true;
    }

    str = getenv("PARSEC_OUTPUT_STDERR_FD");
    if (NULL != str) {
        default_stderr_fd = atoi(str);
    }
    str = getenv("PARSEC_OUTPUT_REDIRECT");
    if (NULL != str) {
        if (0 == strcasecmp(str, "syslog")) {
            parsec_output_redirected_to_syslog = true;
        }
    }
    str = getenv("PARSEC_OUTPUT_SYSLOG_PRI");
#if defined(PARSEC_HAVE_SYSLOG_H)
    if (NULL != str) {
        if (0 == strcasecmp(str, "info")) {
            parsec_output_redirected_syslog_pri = LOG_INFO;
        } else if (0 == strcasecmp(str, "error")) {
            parsec_output_redirected_syslog_pri = LOG_ERR;
        } else if (0 == strcasecmp(str, "warn")) {
            parsec_output_redirected_syslog_pri = LOG_WARNING;
        } else {
            parsec_output_redirected_syslog_pri = LOG_ERR;
        }
    } else {
        parsec_output_redirected_syslog_pri = LOG_ERR;
    }
#endif  /* defined(PARSEC_HAVE_SYSLOG_H) */

    str = getenv("PARSEC_OUTPUT_SYSLOG_IDENT");
    if (NULL != str) {
        redirect_syslog_ident = strdup(str);
    }

    OBJ_CONSTRUCT(&verbose, parsec_output_stream_t);
#if defined(__WINDOWS__)
    {
        WSADATA wsaData;
        WSAStartup( MAKEWORD(2,2), &wsaData );
    }
#endif  /* defined(__WINDOWS__) */
    if (parsec_output_redirected_to_syslog) {
        verbose.lds_want_syslog = true;
        verbose.lds_syslog_priority = parsec_output_redirected_syslog_pri;
        if (NULL != str) {
            verbose.lds_syslog_ident = strdup(redirect_syslog_ident);
        }
        verbose.lds_want_stderr = false;
        verbose.lds_want_stdout = false;
    } else {
        verbose.lds_want_stderr = true;
    }
    gethostname(hostname, sizeof(hostname));
    /* This is spammy and redundant (mpirun --tag-output, etc)
    rc = asprintf(&verbose.lds_prefix, "[%s:%05d] ", hostname, getpid());
    */
    for (i = 0; i < PARSEC_OUTPUT_MAX_STREAMS; ++i) {
        info[i].ldi_used = false;
        info[i].ldi_enabled = false;

        info[i].ldi_syslog = parsec_output_redirected_to_syslog;
        info[i].ldi_file = false;
        info[i].ldi_file_suffix = NULL;
        info[i].ldi_file_want_append = false;
        info[i].ldi_fd = -1;
        info[i].ldi_file_num_lines_lost = 0;
    }

    initialized = true;

    /* Set some defaults */

    rc = asprintf(&output_prefix, "parsec@%s:pid%d.", hostname, getpid());
    output_dir = strdup(parsec_tmp_directory());

    /* Open the default verbose stream */
    verbose_stream = parsec_output_open(&verbose);
    (void)rc;
    return true;
}


/*
 * Open a stream
 */
int parsec_output_open(parsec_output_stream_t * lds)
{
    return do_open(-1, lds);
}


/*
 * Reset the parameters on a stream
 */
int parsec_output_reopen(int output_id, parsec_output_stream_t * lds)
{
    return do_open(output_id, lds);
}


/*
 * Enable and disable output streams
 */
bool parsec_output_switch(int output_id, bool enable)
{
    bool ret = false;

    /* Setup */

    if (!initialized) {
        parsec_output_init();
    }

    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS) {
        ret = info[output_id].ldi_enabled;
        info[output_id].ldi_enabled = enable;
    }

    return ret;
}


/*
 * Reopen all the streams; used during checkpoint/restart.
 */
void parsec_output_reopen_all(void)
{
    int rc;
    char *str;
    char hostname[32];

    str = getenv("PARSEC_OUTPUT_STDERR_FD");
    if (NULL != str) {
        default_stderr_fd = atoi(str);
    } else {
        default_stderr_fd = -1;
    }

    gethostname(hostname, sizeof(hostname));
    if( NULL != verbose.lds_prefix ) {
        free(verbose.lds_prefix);
        verbose.lds_prefix = NULL;
    }
    rc = asprintf(&verbose.lds_prefix, "[%s:%05d] ", hostname, getpid());
#if 0
    int i;
    parsec_output_stream_t lds;

    for (i = 0; i < PARSEC_OUTPUT_MAX_STREAMS; ++i) {

        /* scan till we find ldi_used == 0, which is the end-marker */

        if (!info[i].ldi_used) {
            break;
        }

        /*
         * set this to zero to ensure that parsec_output_open will
         * return this same index as the output stream id
         */
        info[i].ldi_used = false;

#if USE_SYSLOG
        lds.lds_want_syslog = info[i].ldi_syslog;
        lds.lds_syslog_priority = info[i].ldi_syslog_priority;
        lds.lds_syslog_ident = info[i].ldi_syslog_ident;
#else
        lds.lds_want_syslog = false;
#endif
        lds.lds_prefix = info[i].ldi_prefix;
        lds.lds_suffix = info[i].ldi_suffix;
        lds.lds_want_stdout = info[i].ldi_stdout;
        lds.lds_want_stderr = info[i].ldi_stderr;
        lds.lds_want_file = (-1 == info[i].ldi_fd) ? false : true;
        /* open all streams in append mode */
        lds.lds_want_file_append = true;
        lds.lds_file_suffix = info[i].ldi_file_suffix;

        /*
         * call parsec_output_open to open the stream. The return value
         * is guaranteed to be i.  So we can ignore it.
         */
        parsec_output_open(&lds);
    }
#endif
    (void)rc;
}


/*
 * Close a stream
 */
void parsec_output_close(int output_id)
{
    int i;

    /* Setup */

    if (!initialized) {
        return;
    }

    /* If it's valid, used, enabled, and has an open file descriptor,
     * free the resources associated with the descriptor */

    parsec_atomic_lock(&mutex);
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_used && info[output_id].ldi_enabled) {
        free_descriptor(output_id);

        /* If no one has the syslog open, we should close it */

        for (i = 0; i < PARSEC_OUTPUT_MAX_STREAMS; ++i) {
            if (info[i].ldi_used && info[i].ldi_syslog) {
                break;
            }
        }

#if defined(PARSEC_HAVE_SYSLOG)
        if (i >= PARSEC_OUTPUT_MAX_STREAMS && syslog_opened) {
            closelog();
        }
#elif defined(__WINDOWS__)
        if(info[output_id].ldi_syslog_ident != NULL) {
            DeregisterEventSource(info[output_id].ldi_syslog_ident);
        }
#endif
    }

    /* Somewhat of a hack to free up the temp_str */

    if (NULL != temp_str) {
        free(temp_str);
        temp_str = NULL;
        temp_str_len = 0;
    }
    parsec_atomic_unlock(&mutex);
}


/*
 * Main function to send output to a stream
 */
void parsec_output(int output_id, const char *format, ...)
{
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS) {
        va_list arglist;
        va_start(arglist, format);
        output(output_id, format, arglist);
        va_end(arglist);
    }
}


/*
 * Send a message to a stream if the verbose level is high enough
 */
void parsec_output_verbose(int level, int output_id, const char *format, ...)
{
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_verbose_level >= level) {
        va_list arglist;
        va_start(arglist, format);
        output(output_id, format, arglist);
        va_end(arglist);
    }
}


/*
 * Send a message to a stream if the verbose level is high enough
 */
void parsec_output_vverbose(int level, int output_id, const char *format,
                          va_list arglist)
{
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_verbose_level >= level) {
        output(output_id, format, arglist);
    }
}


/*
 * Send a message to a string if the verbose level is high enough
 */
char *parsec_output_string(int level, int output_id, const char *format, ...)
{
    int rc;
    char *ret = NULL;

    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_verbose_level >= level) {
        va_list arglist;
        va_start(arglist, format);
        rc = make_string(&ret, &info[output_id], format, arglist);
        va_end(arglist);
        if (PARSEC_SUCCESS != rc) {
            ret = NULL;
        }
    }

    return ret;
}


/*
 * Send a message to a string if the verbose level is high enough
 */
char *parsec_output_vstring(int level, int output_id, const char *format,
                          va_list arglist)
{
    int rc;
    char *ret = NULL;

    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_verbose_level >= level) {
        rc = make_string(&ret, &info[output_id], format, arglist);
        if (PARSEC_SUCCESS != rc) {
            ret = NULL;
        }
    }

    return ret;
}


/*
 * Set the verbosity level of a stream
 */
void parsec_output_set_verbosity(int output_id, int level)
{
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS) {
        info[output_id].ldi_verbose_level = level;
    }
}


/*
 * Control where output flies will go
 */
void parsec_output_set_output_file_info(const char *dir,
                                      const char *prefix,
                                      char **olddir,
                                      char **oldprefix)
{
    if (NULL != olddir) {
        *olddir = strdup(output_dir);
    }
    if (NULL != oldprefix) {
        *oldprefix = strdup(output_prefix);
    }

    if (NULL != dir) {
        free(output_dir);
        output_dir = strdup(dir);
    }
    if (NULL != prefix) {
        free(output_prefix);
        output_prefix = strdup(prefix);
    }
}


/*
 * Shut down the output stream system
 */
void parsec_output_finalize(void)
{
    if (initialized) {
        if (verbose_stream != -1) {
            parsec_output_close(verbose_stream);
        }
        free(verbose.lds_prefix);
        verbose_stream = -1;

        free (output_prefix);
        free (output_dir);
        OBJ_DESTRUCT(&verbose);
    }
#if defined(__WINDOWS__)
    WSACleanup();
#endif  /* defined(__WINDOWS__) */
}

/************************************************************************/

/*
 * Constructor
 */
static void construct(parsec_object_t *obj)
{
    parsec_output_stream_t *stream = (parsec_output_stream_t*) obj;

    stream->lds_verbose_level = 0;
    stream->lds_syslog_priority = 0;
    stream->lds_syslog_ident = NULL;
    stream->lds_prefix = NULL;
    stream->lds_suffix = NULL;
    stream->lds_is_debugging = false;
    stream->lds_want_syslog = false;
    stream->lds_want_stdout = false;
    stream->lds_want_stderr = false;
    stream->lds_want_file = false;
    stream->lds_want_file_append = false;
    stream->lds_file_suffix = NULL;
}

/*
 * Back-end of open() and reopen().  Necessary to have it as a
 * back-end function so that we can do the thread locking properly
 * (especially upon reopen).
 */
static int do_open(int output_id, parsec_output_stream_t * lds)
{
    int i;

    /* Setup */

    if (!initialized) {
        parsec_output_init();
    }

    /* If output_id == -1, find an available stream, or return
     * PARSEC_ERROR */

    if (-1 == output_id) {
        parsec_atomic_lock(&mutex);
        for (i = 0; i < PARSEC_OUTPUT_MAX_STREAMS; ++i) {
            if (!info[i].ldi_used) {
                break;
            }
        }
        if (i >= PARSEC_OUTPUT_MAX_STREAMS) {
            parsec_atomic_unlock(&mutex);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    /* Otherwise, we're reopening, so we need to free all previous
     * resources, close files, etc. */

    else {
        free_descriptor(output_id);
        i = output_id;
    }

    /* Special case: if we got NULL for lds, then just use the default
     * verbose */

    if (NULL == lds) {
        lds = &verbose;
    }

    /* Got a stream -- now initialize it and open relevant outputs */

    info[i].ldi_used = true;
    if (-1 == output_id) {
        parsec_atomic_unlock(&mutex);
    }
    info[i].ldi_enabled = lds->lds_is_debugging ?
#if defined(PARSEC_DEBUG)
    true : true;
#else
    false : true;
#endif
    info[i].ldi_verbose_level = lds->lds_verbose_level;

#if USE_SYSLOG
#if defined(PARSEC_HAVE_SYSLOG)
    if (parsec_output_redirected_to_syslog) {
        info[i].ldi_syslog = true;
        info[i].ldi_syslog_priority = parsec_output_redirected_syslog_pri;
        if (NULL != redirect_syslog_ident) {
            info[i].ldi_syslog_ident = strdup(redirect_syslog_ident);
            openlog(redirect_syslog_ident, LOG_PID, LOG_USER);
        } else {
            info[i].ldi_syslog_ident = NULL;
            openlog("opal", LOG_PID, LOG_USER);
        }
        syslog_opened = true;
    } else {
#endif
        info[i].ldi_syslog = lds->lds_want_syslog;
        if (lds->lds_want_syslog) {

#if defined(PARSEC_HAVE_SYSLOG)
            if (NULL != lds->lds_syslog_ident) {
                info[i].ldi_syslog_ident = strdup(lds->lds_syslog_ident);
                openlog(lds->lds_syslog_ident, LOG_PID, LOG_USER);
            } else {
                info[i].ldi_syslog_ident = NULL;
                openlog("opal", LOG_PID, LOG_USER);
            }
#elif defined(__WINDOWS__)
            if (NULL == (info[i].ldi_syslog_ident =
                         RegisterEventSource(NULL, TEXT("opal: ")))) {
                /* handle the error */
                return PARSEC_ERROR;
            }
#endif
            syslog_opened = true;
            info[i].ldi_syslog_priority = lds->lds_syslog_priority;
        }

#if defined(PARSEC_HAVE_SYSLOG)
    }
#endif

#else
    info[i].ldi_syslog = false;
#endif

    if (NULL != lds->lds_prefix) {
        info[i].ldi_prefix = strdup(lds->lds_prefix);
        info[i].ldi_prefix_len = (int)strlen(lds->lds_prefix);
    } else {
        info[i].ldi_prefix = NULL;
        info[i].ldi_prefix_len = 0;
    }

    if (NULL != lds->lds_suffix) {
        info[i].ldi_suffix = strdup(lds->lds_suffix);
        info[i].ldi_suffix_len = (int)strlen(lds->lds_suffix);
    } else {
        info[i].ldi_suffix = NULL;
        info[i].ldi_suffix_len = 0;
    }

    if (parsec_output_redirected_to_syslog) {
        /* since all is redirected to syslog, ensure
         * we don't duplicate the output to the std places
         */
        info[i].ldi_stdout = false;
        info[i].ldi_stderr = false;
        info[i].ldi_file = false;
        info[i].ldi_fd = -1;
    } else {
        /* since we aren't redirecting, use what was
         * given to us
         */
        info[i].ldi_stdout = lds->lds_want_stdout;
        info[i].ldi_stderr = lds->lds_want_stderr;

        info[i].ldi_fd = -1;
        info[i].ldi_file = lds->lds_want_file;
        info[i].ldi_file_suffix = (NULL == lds->lds_file_suffix) ? NULL :
            strdup(lds->lds_file_suffix);
        info[i].ldi_file_want_append = lds->lds_want_file_append;
        info[i].ldi_file_num_lines_lost = 0;
    }

    /* Don't open a file in the session directory now -- do that lazily
     * so that if there's no output, we don't have an empty file */

    return i;
}


static int open_file(int i)
{
    int flags;
    char *filename;

    /* Setup the filename and open flags */

    if (NULL != output_dir) {
        filename = (char *) malloc(MAXPATHLEN);
        if (NULL == filename) {
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        strncpy(filename, output_dir, MAXPATHLEN);
        strcat(filename, "/");
        if (NULL != output_prefix) {
            strcat(filename, output_prefix);
        }
        if (info[i].ldi_file_suffix != NULL) {
            strcat(filename, info[i].ldi_file_suffix);
        } else {
            info[i].ldi_file_suffix = NULL;
            strcat(filename, "output.txt");
        }
        flags = O_CREAT | O_RDWR;
        if (!info[i].ldi_file_want_append) {
            flags |= O_TRUNC;
        }

        /* Actually open the file */
        info[i].ldi_fd = open(filename, flags, 0644);
        if (-1 == info[i].ldi_fd) {
            info[i].ldi_used = false;
            free(filename);
            return PARSEC_ERROR;
        }

        free(filename);

        /* Make the file be close-on-exec to prevent child inheritance
         * problems */

#ifndef __WINDOWS__
        /* TODO: Need to find out the equivalent in windows */
        if (-1 == fcntl(info[i].ldi_fd, F_SETFD, 1)) {
           return PARSEC_ERROR;
        }
#endif

    }

    /* Return successfully even if the session dir did not exist yet;
     * we'll try opening it later */

    return PARSEC_SUCCESS;
}


/*
 * Free all the resources associated with a descriptor.
 */
static void free_descriptor(int output_id)
{
    output_desc_t *ldi;

    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_used && info[output_id].ldi_enabled) {
        ldi = &info[output_id];

        if (-1 != ldi->ldi_fd) {
            close(ldi->ldi_fd);
        }
        ldi->ldi_used = false;

        /* If we strduped a prefix, suffix, or syslog ident, free it */

        if (NULL != ldi->ldi_prefix) {
            free(ldi->ldi_prefix);
        }
        ldi->ldi_prefix = NULL;

    if (NULL != ldi->ldi_suffix) {
        free(ldi->ldi_suffix);
    }
    ldi->ldi_suffix = NULL;

    if (NULL != ldi->ldi_file_suffix) {
            free(ldi->ldi_file_suffix);
        }
        ldi->ldi_file_suffix = NULL;

#ifndef __WINDOWS__
        if (NULL != ldi->ldi_syslog_ident) {
            free(ldi->ldi_syslog_ident);
        }
        ldi->ldi_syslog_ident = NULL;
#endif
    }
}


static int make_string(char **no_newline_string, output_desc_t *ldi,
                       const char *format, va_list arglist)
{
    int rc;
    size_t len, total_len;
    bool want_newline = false;

    /* Make the formatted string */
    rc = vasprintf(no_newline_string, format, arglist);
    total_len = len = strlen(*no_newline_string);
    if ('\n' != (*no_newline_string)[len - 1]) {
        want_newline = true;
        ++total_len;
    } else if (NULL != ldi->ldi_suffix) {
        /* if we have a suffix, then we don't want a
         * newline to appear before it
         */
        (*no_newline_string)[len - 1] = '\0';
        want_newline = true; /* add newline to end after suffix */
        /* total_len won't change since we just moved the newline
         * to appear after the suffix
         */
    }
    if (NULL != ldi->ldi_prefix) {
        total_len += strlen(ldi->ldi_prefix);
    }
    if (NULL != ldi->ldi_suffix) {
        total_len += strlen(ldi->ldi_suffix);
    }
    if (temp_str_len < total_len + want_newline) {
        if (NULL != temp_str) {
            free(temp_str);
        }
        temp_str = (char *) malloc(total_len * 2);
        if (NULL == temp_str) {
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        temp_str_len = total_len * 2;
    }
    if (NULL != ldi->ldi_prefix && NULL != ldi->ldi_suffix) {
        if (want_newline) {
            snprintf(temp_str, temp_str_len, "%s%s%s\n",
                     ldi->ldi_prefix, *no_newline_string, ldi->ldi_suffix);
        } else {
            snprintf(temp_str, temp_str_len, "%s%s%s", ldi->ldi_prefix,
                     *no_newline_string, ldi->ldi_suffix);
        }
    } else if (NULL != ldi->ldi_prefix) {
        if (want_newline) {
            snprintf(temp_str, temp_str_len, "%s%s\n",
                     ldi->ldi_prefix, *no_newline_string);
        } else {
            snprintf(temp_str, temp_str_len, "%s%s", ldi->ldi_prefix,
                     *no_newline_string);
        }
    } else if (NULL != ldi->ldi_suffix) {
        if (want_newline) {
            snprintf(temp_str, temp_str_len, "%s%s\n",
                     *no_newline_string, ldi->ldi_suffix);
        } else {
            snprintf(temp_str, temp_str_len, "%s%s",
                     *no_newline_string, ldi->ldi_suffix);
        }
    } else {
        if (want_newline) {
            snprintf(temp_str, temp_str_len, "%s\n", *no_newline_string);
        } else {
            snprintf(temp_str, temp_str_len, "%s", *no_newline_string);
        }
    }

    (void)rc;
    return PARSEC_SUCCESS;
}

/*
 * Do the actual output.  Take a va_list so that we can be called from
 * multiple different places, even functions that took "..." as input
 * arguments.
 */
static int output(int output_id, const char *format, va_list arglist)
{
    int nbwrote, rc = PARSEC_SUCCESS;
    char *str, *out = NULL;
    output_desc_t *ldi;

    /* Setup */

    if (!initialized) {
        parsec_output_init();
    }

    /* If it's valid, used, and enabled, output */

    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS &&
        info[output_id].ldi_used && info[output_id].ldi_enabled) {
        parsec_atomic_lock(&mutex);
        ldi = &info[output_id];

        /* Make the strings */
        if (PARSEC_SUCCESS != (rc = make_string(&str, ldi, format, arglist))) {
            parsec_atomic_unlock(&mutex);
            return rc;
        }

        /* Syslog output -- does not use the newline-appended string */
#if defined(PARSEC_HAVE_SYSLOG)
        if (ldi->ldi_syslog) {
            syslog(ldi->ldi_syslog_priority, "%s", str);
        }
#endif

        /* All others (stdout, stderr, file) use temp_str, potentially
           with a newline appended */

        out = temp_str;

        /* stdout output */
        if (ldi->ldi_stdout) {
            nbwrote = write(fileno(stdout), out, (int)strlen(out));
            fflush(stdout);
        }

        /* stderr output */
        if (ldi->ldi_stderr) {
            nbwrote = write((-1 == default_stderr_fd) ?
                            fileno(stderr) : default_stderr_fd,
                            out, (int)strlen(out));
            fflush(stderr);
        }

        /* File output -- first check to see if the file opening was
         * delayed.  If so, try to open it.  If we failed to open it,
         * then just discard (there are big warnings in the
         * parsec_output.h docs about this!). */

        if (ldi->ldi_file) {
            if (ldi->ldi_fd == -1) {
                if (PARSEC_SUCCESS != open_file(output_id)) {
                    ++ldi->ldi_file_num_lines_lost;
                } else if (ldi->ldi_file_num_lines_lost > 0) {
                    char buffer[BUFSIZ];
                    memset(buffer, 0, BUFSIZ);
                    snprintf(buffer, BUFSIZ - 1,
                             "[WARNING: %d lines lost because the Open MPI process session directory did\n not exist when parsec_output() was invoked]\n",
                             ldi->ldi_file_num_lines_lost);
                    assert(ldi->ldi_fd >= 0);
                    nbwrote = write(ldi->ldi_fd, buffer, (int)strlen(buffer));
                    ldi->ldi_file_num_lines_lost = 0;
                }
            }
            if (ldi->ldi_fd != -1) {
                nbwrote = write(ldi->ldi_fd, out, (int)strlen(out));
            }
        }
        parsec_atomic_unlock(&mutex);
        free(str);
    }

    (void)nbwrote;
    return rc;
}

int parsec_output_get_verbosity(int output_id)
{
    if (output_id >= 0 && output_id < PARSEC_OUTPUT_MAX_STREAMS && info[output_id].ldi_used) {
        return info[output_id].ldi_verbose_level;
    } else {
        return -1;
    }
}


#if !defined(PARSEC_HAVE_VASPRINTF)
#include <errno.h>

int vasprintf(char **ptr, const char *fmt, va_list ap) {
    int length;
    va_list ap2;
    char dummy[4];

    /* va_list might have pointer to internal state and using
       it twice is a bad idea.  So make a copy for the second
       use.  Copy order taken from Autoconf docs. */
#if defined(PARSEC_HAVE_VA_COPY)
    va_copy(ap2, ap);
#elif defined(PARSEC_HAVE_UNDERSCORE_VA_COPY)
    __va_copy(ap2, ap);
#else
    memcpy (&ap2, &ap, sizeof(va_list));
#endif

    /* guess the size using a nice feature of snprintf and friends:
     *
     *  The functions snprintf() and vsnprintf() do not write more than size bytes (including
     *  the  trailing  '\0').  If the output was truncated due to this limit then the return
     *  value is the number of characters (not including the trailing '\0') which  would
     *  have  been written  to  the  final  string  if enough space had been available.
     */
    length = vsnprintf(dummy, 4, fmt, ap2);

#if defined(PARSEC_HAVE_VA_COPY) || defined(PARSEC_HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(PARSEC_HAVE_VA_COPY) || defined(PARSEC_HAVE_UNDERSCORE_VA_COPY) */

    /* allocate a buffer */
    *ptr = (char *) malloc((size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        return -1;
    }

    /* fill the buffer */
    length = vsprintf(*ptr, fmt, ap);


    /* realloc */
    *ptr = (char*) realloc(*ptr, (size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        return -1;
    }

    return length;
}
#endif  /* !defined(PARSEC_HAVE_VASPRINTF) */

#if !defined(PARSEC_HAVE_ASPRINTF)
int asprintf(char **ptr, const char *fmt, ...) {
    int length;
    va_list ap;

    va_start(ap, fmt);
    length = vasprintf(ptr, fmt, ap);
    va_end(ap);

    return length;
}
#endif  /* !defined(PARSEC_HAVE_ASPRINTF) */
