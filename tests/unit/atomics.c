#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#include "parsec/sys/atomic.h"
#include "parsec/bindthread.h"
#include "parsec/class/barrier.h"

typedef struct {
    int32_t      mo32;
    int32_t      ma32;
    int64_t      mo64;
    int64_t      ma64;

    int32_t      i32;
    int32_t      padding1;
    int64_t      i64;
    
#if defined(PARSEC_HAVE_INT128)
    __int128_t   i128;
    __int128_t   mo128;
    __int128_t   ma128;
#endif
} values_t;

typedef struct {
    int            nb_tests;
    int            thid;
    unsigned short xsubi[3];

    values_t      *lv;
    values_t      *gv;
} param_t;

static parsec_barrier_t barrier1;

static void *pfunction(void *_param)
{
    int i;
    int r;
    int32_t l16_32, l32;
    int64_t l32_64, l64;
#if defined(PARSEC_HAVE_INT128)
    __int128_t l64_128, l128;
    int nb_cases = 18;
#else
    int nb_cases = 14;
#endif
    param_t *param = (param_t*)_param;
    
    parsec_bindthread(param->thid, 0);

    parsec_barrier_wait(&barrier1);

    for(i = 0; i < param->nb_tests; i++) {
        r = (int)( (double)nb_cases * erand48(param->xsubi) );
        l32 = nrand48(param->xsubi);
        l16_32 = l32 & 0xFFFF;
        l64 = (((int64_t)nrand48(param->xsubi)) << 32) | l32;
        l32_64 = l64 & 0xFFFFFFFF;
#if defined(PARSEC_HAVE_INT128)
        l128 = nrand48(param->xsubi);
        l128 = (l128 << 32) | nrand48(param->xsubi);
        l128 = (l128 << 32) | nrand48(param->xsubi);
        l128 = (l128 << 32) | nrand48(param->xsubi);
        l64_128 = l128 & 0xFFFFFFFFFFFFFFFF;;
#endif
        switch(r) {
        case 0:
            param->lv->mo32 |= l32;
            parsec_atomic_fetch_or_int32(&param->gv->mo32, l32);
            break;
        case 1:
            param->lv->mo64 |= l64;
            parsec_atomic_fetch_or_int64(&param->gv->mo64, l64);
            break;
        case 2:
            param->lv->ma32 &= l32;
            parsec_atomic_fetch_and_int32(&param->gv->ma32, l32);
            break;
        case 3:
            param->lv->ma64 &= l64;
            parsec_atomic_fetch_and_int64(&param->gv->ma64, l64);
            break;
        case 4:
            param->lv->i32 += l16_32;
            parsec_atomic_fetch_add_int32(&param->gv->i32, l16_32);
            break;
        case 5:
            param->lv->i32++;
            parsec_atomic_fetch_inc_int32(&param->gv->i32);
            break;
        case 6:
            param->lv->i32 -= l16_32;
            parsec_atomic_fetch_sub_int32(&param->gv->i32, l16_32);
            break;
        case 7:
            param->lv->i32--;
            parsec_atomic_fetch_dec_int32(&param->gv->i32);
            break;
        case 8:
            param->lv->i64 += l32_64;
            parsec_atomic_fetch_add_int64(&param->gv->i64, l32_64);
            break;
        case 9:
            param->lv->i64++;
            parsec_atomic_fetch_inc_int64(&param->gv->i64);
            break;
        case 10:
            param->lv->i64 -= l32_64;
            parsec_atomic_fetch_sub_int64(&param->gv->i64, l32_64);
            break;
        case 11:
            param->lv->i64--;
            parsec_atomic_fetch_dec_int64(&param->gv->i64);
            break;
#if defined(PARSEC_HAVE_INT128)
        case 12:
            param->lv->mo128 |= l128;
            parsec_atomic_fetch_or_int128(&param->gv->mo128, l128);
            break;
        case 13:
            param->lv->ma128 &= l128;
            parsec_atomic_fetch_and_int128(&param->gv->ma128, l128);
            break;
        case 14:
            param->lv->i128 += l64_128;
            parsec_atomic_fetch_add_int128(&param->gv->i128, l64_128);
            break;
        case 15:
            param->lv->i128++;
            parsec_atomic_fetch_inc_int128(&param->gv->i128);
            break;
        case 16:
            param->lv->i128 -= l64_128;
            parsec_atomic_fetch_sub_int128(&param->gv->i128, l64_128);
            break;
        case 17:
            param->lv->i128--;
            parsec_atomic_fetch_dec_int128(&param->gv->i128);
            break;
#endif
        default:
            break;
        }
    }

    return NULL;
}

#if defined(PARSEC_HAVE_INT128)
static void positive_int128_tostr_rec(__uint128_t n, char *out, int offset, int base) {
    if (n == 0) {
        out[offset] = '\0';
      return;
    }
    positive_int128_tostr_rec(n/base, out, offset+1, base);
    if( base == 10 )
        out[offset]= (char)((n % 10) + '0');
    if( base == 16 ) {
        char v[2];
        sprintf(v, "%x", (unsigned int)(n % 16));
        out[offset] = v[0];
    }
}

static char *int128_to_str_base10(__int128_t n, char *out) {
    if( n < 0 ) {
        out[0] = '-';
        positive_int128_tostr_rec(-n, out+1, 0, 10);
    }
    if( n > 0 ) {
        positive_int128_tostr_rec(n, out, 0, 10);
    }
    if( n == 0 ) {
        out[0] = '0';
        out[1] = '\0';
    }
    return out;
}

static char *int128_to_str_base16(__int128_t n, char *out) {
    if( n != 0 )
        positive_int128_tostr_rec( (__uint128_t)n, out, 0, 16 );
    else {
        out[0] = '0';
        out[1] = '\0';        
    }
    return out;
}
#endif

int main(int argc, char *argv[])
{
    int nb_threads = 8;
    int nb_tests = 8192;
    pthread_t *threads;
    param_t *params;
    int ch;
    char *m;
    values_t *values;
    unsigned short xsubi[3];
    int i, ret = 0;
    struct timeval now;
#if defined(PARSEC_HAVE_INT128)
    char v128a[128];
    char v128b[128];
#endif
    
    while( (ch = getopt(argc, argv, "c:n:h?")) != -1 ) {
        switch(ch) {
        case 'c':
            nb_threads = strtol(optarg, &m, 0);
            if( (nb_threads < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -c value");
                exit(1);
            }
            break;
        case 'n':
            nb_tests = strtol(optarg, &m, 0);
            if( (nb_tests < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -n value");
                exit(1);
            }
            break;
        case '?':
        case 'h':
        default :
            fprintf(stderr, "Usage: %s [-c nbthreads|-n nbtests]\n", argv[0]);
            exit(1);
            break;
        }
    }

    threads = calloc(sizeof(pthread_t), nb_tests);
    params = calloc(sizeof(param_t), nb_tests);
    values = calloc(sizeof(values_t), nb_tests+2);

    gettimeofday(&now, NULL);
    srand48(now.tv_usec ^ getpid());
    xsubi[0] = lrand48();
    xsubi[1] = lrand48();
    xsubi[2] = lrand48();

    /* values[nb_threads] is the initial value */
    values[nb_threads].mo32 = nrand48(xsubi);
    values[nb_threads].ma32 = nrand48(xsubi);
    values[nb_threads].mo64 = nrand48(xsubi);
    values[nb_threads].mo64 = (values[nb_threads].mo64 << 32) | nrand48(xsubi);
    values[nb_threads].ma64 = nrand48(xsubi);
    values[nb_threads].ma64 = (values[nb_threads].ma64 << 32) | nrand48(xsubi);
    values[nb_threads].i32 = nrand48(xsubi) & 0xFFFF;
    values[nb_threads].i64 = nrand48(xsubi);
#if defined(PARSEC_HAVE_INT128)
    values[nb_threads].i128 = nrand48(xsubi);
    values[nb_threads].i128 = (values[nb_threads].i128 << 32) | nrand48(xsubi);
    values[nb_threads].mo128 = nrand48(xsubi);
    values[nb_threads].mo128 = (values[nb_threads].mo128 << 32) | nrand48(xsubi);
    values[nb_threads].mo128 = (values[nb_threads].mo128 << 32) | nrand48(xsubi);
    values[nb_threads].mo128 = (values[nb_threads].mo128 << 32) | nrand48(xsubi);
    values[nb_threads].ma128 = nrand48(xsubi);
    values[nb_threads].ma128 = (values[nb_threads].ma128 << 32) | nrand48(xsubi);
    values[nb_threads].ma128 = (values[nb_threads].ma128 << 32) | nrand48(xsubi);
    values[nb_threads].ma128 = (values[nb_threads].ma128 << 32) | nrand48(xsubi);
#endif
    /* We keep a copy of the initial value in values[nb_threads+1] */
    memcpy(&values[nb_threads+1], &values[nb_threads], sizeof(values_t));

    /* We initialize local values[i] with the neutral element of each operation
     * so that we can 'reproduce' the result using associativity and commutativity */
    for(i = 1; i < nb_threads; i++) {
        values[i].mo32 = 0;
        values[i].ma32 = 0; values[i].ma32 = ~values[i].ma32;
        values[i].mo64 = 0;
        values[i].ma64 = 0; values[i].ma64 = ~values[i].ma64;
        values[i].i32  = 0;
        values[i].i64  = 0;
#if defined(PARSEC_HAVE_INT128)
        values[i].i128 = 0;
        values[i].mo128 = 0;
        values[i].ma128 = 0; values[i].ma128 = ~values[i].ma128;
#endif
    }

    for(i = 0; i < nb_threads; i++) {
        params[i].nb_tests = nb_tests;
        params[i].thid = i;
        params[i].xsubi[0] = lrand48();
        params[i].xsubi[1] = lrand48();
        params[i].xsubi[2] = lrand48();
        params[i].lv = &values[i];
        params[i].gv = &values[nb_threads];
    }
    parsec_barrier_init(&barrier1, NULL, nb_threads);
    for(i = 1; i < nb_threads; i++) {
        pthread_create(&threads[i], NULL, pfunction, &params[i]);
    }
    pfunction( &params[0] );
    for(i = 1; i < nb_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    for(i = 0; i < nb_threads; i++) {
        values[nb_threads+1].mo32 |= values[i].mo32;
        values[nb_threads+1].ma32 &= values[i].ma32;
        values[nb_threads+1].mo64 |= values[i].mo64;
        values[nb_threads+1].ma64 &= values[i].ma64;
        values[nb_threads+1].i32  += values[i].i32;
        values[nb_threads+1].i64  += values[i].i64;
#if defined(PARSEC_HAVE_INT128)
        values[nb_threads+1].i128 += values[i].i128;
        values[nb_threads+1].mo128 |= values[i].mo128;
        values[nb_threads+1].ma128 &= values[i].ma128;
#endif
    }

    if( values[nb_threads+1].mo32 != values[i].mo32 ) {
        fprintf(stderr, "Error in Mask Or operation on 32 bits: expected %08x, got %08x\n",
                values[nb_threads+1].mo32, values[i].mo32);
        ret++;
    } else {
        fprintf(stderr, "No error in Mask Or Operation on 32 bits: got %08x\n",
                values[nb_threads+1].mo32);
    }
    if( values[nb_threads+1].ma32 != values[i].ma32 ) {
        fprintf(stderr, "Error in Mask And operation on 32 bits: expected %08x, got %08x\n",
                values[nb_threads+1].ma32, values[i].ma32);
        ret++;
    } else {
        fprintf(stderr, "No error in Mask And Operation on 32 bits: got %08x\n",
                values[nb_threads+1].ma32);
    }
    if( values[nb_threads+1].mo64 != values[i].mo64 ) {
        fprintf(stderr, "Error in Mask Or operation on 64 bits: expected %16lx, got %16lx\n",
                values[nb_threads+1].mo64, values[i].mo64);
        ret++;
    } else {
        fprintf(stderr, "No error in Mask Or Operation on 64 bits: got %16lx\n",
                values[nb_threads+1].mo64);
    }
    if( values[nb_threads+1].ma64 != values[i].ma64 ) {
        fprintf(stderr, "Error in Mask And operation on 64 bits: expected %16lx, got %16lx\n",
                values[nb_threads+1].ma64, values[i].ma64);
        ret++;
    } else {
        fprintf(stderr, "No error in Mask And Operation on 64 bits: got %16lx\n",
                values[nb_threads+1].ma64);
    }
    if( values[nb_threads+1].i32 != values[i].i32 ) {
        fprintf(stderr, "Error in integer operations on 32 bits: expected %08x, got %08x\n",
                values[nb_threads+1].i32, values[i].i32);
        ret++;
    } else {
        fprintf(stderr, "No error in integer operation on 32 bits: got %d\n",
                values[nb_threads+1].i32);
    }
    if( values[nb_threads+1].i64 != values[i].i64 ) {
        fprintf(stderr, "Error in integer operations on 64 bits: expected %ld, got %ld\n",
                values[nb_threads+1].i64, values[i].i64);
        ret++;
    } else {
        fprintf(stderr, "No error in integer operation on 64 bits: got %ld\n",
                values[nb_threads+1].i64);
    }
#if defined(PARSEC_HAVE_INT128)
    if( values[nb_threads+1].i128 != values[i].i128 ) {
        fprintf(stderr, "Error in integer operations on 128 bits: expected %s, got %s\n",
                int128_to_str_base10(values[nb_threads+1].i128, v128a),
                int128_to_str_base10(values[i].i128, v128b));
        ret++;
    } else {
        fprintf(stderr, "No error in integer operation on 128 bits: got %s\n",
                int128_to_str_base10(values[nb_threads+1].i128, v128a));
    }
    if( values[nb_threads+1].mo128 != values[i].mo128 ) {
        fprintf(stderr, "Error in Mask Or operation on 128 bits: expected %s, got %s\n",
                int128_to_str_base16(values[nb_threads+1].mo128, v128a),
                int128_to_str_base16(values[i].mo128, v128b));
        ret++;
    } else {
        fprintf(stderr, "No error in Mask Or Operation on 128 bits: got %s\n",
                int128_to_str_base16(values[nb_threads+1].mo128, v128a));
    }
    if( values[nb_threads+1].ma128 != values[i].ma128 ) {
        fprintf(stderr, "Error in Mask And operation on 128 bits: expected %s, got %s\n",
                int128_to_str_base16(values[nb_threads+1].ma128, v128a),
                int128_to_str_base16(values[i].ma128, v128b));
        ret++;
    } else {
        fprintf(stderr, "No error in Mask And Operation on 128 bits: got %s\n",
                int128_to_str_base16(values[nb_threads+1].ma128, v128a));
    }
#endif

    return ret;
}
