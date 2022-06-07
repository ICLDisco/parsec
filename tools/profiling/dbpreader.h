/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dbpreader_h_
#define _dbpreader_h_

#include "parsec/parsec_config.h"

/* Basic Key-Value Info interface */

typedef struct dbp_file dbp_file_t;
typedef struct dbp_info dbp_info_t;
char *dbp_info_get_key(const dbp_info_t *info);
char *dbp_info_get_value(const dbp_info_t *info);

/* Multifile DBP reader */

typedef struct dbp_multifile_reader dbp_multifile_reader_t;
dbp_multifile_reader_t *dbp_reader_open_files(int nbfiles, char *files[]);
int dbp_reader_nb_files(const dbp_multifile_reader_t *dbp);
int dbp_reader_nb_dictionary_entries(const dbp_multifile_reader_t *dbp);
int dbp_reader_last_error(const dbp_multifile_reader_t *dbp);
void dbp_reader_close_files(dbp_multifile_reader_t *dbp);
void dbp_reader_destruct(dbp_multifile_reader_t *dbp);

/* Dictionary interface */

typedef struct dbp_dictionary dbp_dictionary_t;
dbp_dictionary_t *dbp_file_get_dictionary(const dbp_file_t *dbp, int did);
dbp_dictionary_t *dbp_reader_get_dictionary(const dbp_multifile_reader_t *dbp, int did);
int dbp_file_translate_local_dico_to_global(const dbp_file_t *file, int lid);
char *dbp_dictionary_name(const dbp_dictionary_t *dico);
char *dbp_dictionary_attributes(const dbp_dictionary_t *dico);
char *dbp_dictionary_convertor(const dbp_dictionary_t *dico);
int dbp_dictionary_keylen(const dbp_dictionary_t *dico);

/* Single DBP file interface */

//typedef struct dbp_file dbp_file_t;
dbp_file_t *dbp_reader_get_file(const dbp_multifile_reader_t *dbp, int fid);

char * dbp_file_hr_id(const dbp_file_t *file);
char * dbp_file_get_name(const dbp_file_t *file);
int dbp_file_get_rank(const dbp_file_t *file);
int dbp_file_nb_threads(const dbp_file_t *file);
int dbp_file_nb_infos(const dbp_file_t *file);
int dbp_file_nb_dictionary_entries(const dbp_file_t *file);
int dbp_file_error(const dbp_file_t *file);
dbp_info_t *dbp_file_get_info(const dbp_file_t *file, int iid);

/* Single DBP thread interface */

typedef struct dbp_thread dbp_thread_t;
dbp_thread_t *dbp_file_get_thread(const dbp_file_t *file, int tid);

int dbp_thread_nb_events(const dbp_thread_t *th);
int dbp_thread_nb_infos(const dbp_thread_t *th);
char *dbp_thread_get_hr_id(const dbp_thread_t *th);
dbp_info_t *dbp_thread_get_info(const dbp_thread_t *th, int iid);

/* Events iteration */
typedef struct dbp_event dbp_event_t;

typedef struct dbp_event_iterator dbp_event_iterator_t;
dbp_event_iterator_t *dbp_iterator_new_from_thread(const dbp_thread_t *th);
dbp_event_iterator_t *dbp_iterator_new_from_iterator(const dbp_event_iterator_t *it);
const dbp_event_t *dbp_iterator_current(dbp_event_iterator_t *it);
const dbp_event_t *dbp_iterator_first(dbp_event_iterator_t *it);
const dbp_event_t *dbp_iterator_next(dbp_event_iterator_t *it);
void dbp_iterator_delete(dbp_event_iterator_t *it);
int dbp_iterator_move_to_matching_event(dbp_event_iterator_t *pos, const dbp_event_t *ref);
dbp_event_iterator_t *dbp_iterator_find_matching_event_all_threads(const dbp_event_iterator_t *pos);
const dbp_thread_t *dbp_iterator_thread(const dbp_event_iterator_t *it);

int dbp_event_get_key(const dbp_event_t *e);
int dbp_event_get_flags(const dbp_event_t *e);
uint64_t dbp_event_get_event_id(const dbp_event_t *e);
uint32_t dbp_event_get_taskpool_id(const dbp_event_t *e);
uint64_t dbp_event_get_timestamp(const dbp_event_t *e);
void *dbp_event_get_info(const dbp_event_t *e);
int   dbp_event_info_len(const dbp_event_t *e, const dbp_file_t *dbp);

// DEBUG
void dbp_file_print(dbp_file_t * file);

#endif /* _dbpreader_h_ */
