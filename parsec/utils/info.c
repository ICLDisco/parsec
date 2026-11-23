#include "parsec_config.h"
#include "parsec.h"
#include "parsec/utils/info.h"
#include "parsec/class/hashtable.h"
#include "parsec/class/parsec_object.h"

PARSEC_OBJ_CLASS_INSTANCE(parsec_info_t, parsec_hashtable_t, NULL, NULL);

int parsec_info_set(parsec_info_t *info, const char *key, char *value) {
    return PARSEC_NOT_IMPLEMENTED;
}

int parsec_info_get(parsec_info_t *info, const char *key, char **value) {
    return PARSEC_NOT_IMPLEMENTED;
}

int parsec_info_clear(parsec_info_t *info) {
    /* must clear in-place as multiple refs ot the info may be held */
    return PARSEC_NOT_IMPLEMENTED;
}

int parsec_context_get_info(parsec_context_t *ctx, parsec_info_t **info) {
    if(NULL == ctx->info) {
        ctx->info = PARSEC_OBJ_NEW(parsec_info_t);
    }
    return ctx->info;
}

int parsec_taskpool_get_info(parsec_taskpool_t *tp, parsec_info_t **info) {
    if(NULL == tp->info) {
        tp->info = PARSEC_OBJ_NEW(parsec_info_t);
    }
    return tp->info;
}
