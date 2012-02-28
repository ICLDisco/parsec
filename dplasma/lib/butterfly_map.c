#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "dplasma/lib/butterfly_map.h"

static int segment_to_type_index(seg_info_t seg, int m, int n);

seg_info_t dague_rbt_calculate_constants(int N, int nb, int L, int ib, int jb){
    int am, an, bm, bn, cm, cn, dm, dn, em, en, fm, fn;
    int mb, width, height, block_count;
    seg_info_t seg;

    memset(&seg, 0, sizeof(seg_info_t));

    /* The matrix has to be symmetric if we are applying the random butterfly transformation */
    mb = nb;
    block_count = 1<<L;

    assert(ib>=jb);

    /* Calculate starting, middle and ending point for this butterfly */
    seg.spm = ib*N/block_count;
    seg.spn = jb*N/block_count;

    seg.epm = (ib+1)*N/block_count-1;
    seg.epn = (jb+1)*N/block_count-1;

    seg.mpm = (seg.spm + seg.epm + 1)/2;
    seg.mpn = (seg.spn + seg.epn + 1)/2;

    /* Calculate the different sizes that might appear */
    am = seg.spm%nb;
    bm = mb-am;
    cm = seg.mpm%nb;
    dm = mb-cm;
    em = (dm>bm) ? dm-bm : nb + (dm-bm);
    fm = mb-em;

    an = seg.spn%nb;
    bn = mb-an;
    cn = seg.mpn%nb;
    dn = mb-cn;
    en = (dn>bn) ? dn-bn : mb + (dn-bn);
    fn = mb-en;

    /* top edge types */
    if( bm < fm+em ) {
        if( fm < bm ) {
            height = bm-fm;
            seg.t_cnt.m = 2;
            seg.t_sz.m1 = height;
            seg.t_sz.m2 = fm;
        }else{
            height = bm;
            seg.t_cnt.m = 1;
            seg.t_sz.m1 = height;
            seg.t_sz.m2 = 0;
        }
    }

    /* left edge types */
    if( bn < fn+en ) {
        if( fn < bn ) {
            width = bn-fn;
            seg.l_cnt.n = 2;
            seg.l_sz.n1 = width;
            seg.l_sz.n2 = fn;
        }else{
            width = bn;
            seg.l_cnt.n = 1;
            seg.l_sz.n1 = width;
            seg.l_sz.n2 = 0;
        }
    }

    /* right edge types */
    if( cn < en+fn && cn ) {
        if( en < cn ){
            width = cn-en;
            seg.r_cnt.n = 2;
            seg.r_sz.n1 = en;
            seg.r_sz.n2 = width;
        }else{
            width = cn;
            seg.r_cnt.n = 1;
            seg.r_sz.n1 = width;
            seg.r_sz.n2 = 0;
        }
    }

    /* bottom edge types */
    if( cm < em+fm && cm) {
        if( em < cm ){
            height = cm-em;
            seg.b_cnt.m = 2;
            seg.b_sz.m1 = em;
            seg.b_sz.m2 = height;
        }else{
            height = cm;
            seg.b_cnt.m = 1;
            seg.b_sz.m1 = height;
            seg.b_sz.m2 = 0;
        }
    }

    /* center types */
    do{
        int startn, endn;

        if( 0 < fn ){
            seg.c_cnt.n = 2;
            seg.c_cnt.m = 2;
        }else{
            seg.c_cnt.n = 1;
            seg.c_cnt.m = 1;
        }
        seg.c_sz.n1 = en;
        seg.c_sz.n2 = fn;
        seg.c_sz.m1 = em;
        seg.c_sz.m2 = fm;

        startn = seg.spn;
        if( bn != nb ){
          startn += bn;
        }
        endn = seg.mpn;
        if( cn != nb ){
            endn -= cn;
        }

        seg.c_seg_cnt = seg.c_cnt.n*(endn-startn)/nb;
    }while(0); // just to give me a scope without looking ugly.

    seg.tot_seg_cnt_n = 2*(seg.l_cnt.n + seg.c_seg_cnt + seg.r_cnt.n);
    seg.tot_seg_cnt_m = 2*(seg.t_cnt.m + seg.c_seg_cnt + seg.b_cnt.m);

    return seg;
}

void segment_to_tile(dague_seg_ddesc_t *seg_ddesc, int m, int n, int *m_tile, int *n_tile, uintptr_t *offset){
    seg_info_t seg;
    int mb, nb;
    int abs_m, abs_n;
    int right=0, bottom=0;

    seg = seg_ddesc->seg_info;
    mb = seg_ddesc->A_org->mb;
    nb = seg_ddesc->A_org->nb;

    if( n >= seg.tot_seg_cnt_n || m >= seg.tot_seg_cnt_m ){
        fprintf(stderr,"invalid segment coordinates\n");
        return;
    }

    if( n >= seg.tot_seg_cnt_n/2 ){
        n -= seg.tot_seg_cnt_n/2;
        right = 1;
    }
    if( m >= seg.tot_seg_cnt_m/2 ){
        m -= seg.tot_seg_cnt_m/2;
        bottom = 1;
    }

    /* Horizontal */
    if( n < seg.l_cnt.n ){ /* left edge */
        abs_n = seg.spn;
        if( 1 == n ){
            abs_n += seg.l_sz.n1;
        }
    }else if( n < (seg.l_cnt.n+seg.c_seg_cnt) ){ /* center */
        abs_n = seg.spn + seg.l_sz.n1 + seg.l_sz.n2;
        abs_n += ((n-seg.l_cnt.n)/seg.c_cnt.n)*nb;
        if( (n-seg.l_cnt.n) % seg.c_cnt.n ){
            abs_n += seg.c_sz.n1;
        }
    }else{ /* right edge */
        abs_n = seg.mpn - (seg.r_sz.n1 + seg.r_sz.n2);
        if( n - (seg.l_cnt.n+seg.c_seg_cnt) ){
            abs_n += seg.r_sz.n1;
        }
    }

    /* Vertical */
    if( m < seg.t_cnt.m ){ /* top edge */
        abs_m = seg.spm;
        if( 1 == m ){
            abs_m += seg.t_sz.m1;
        }
    }else if( m < (seg.t_cnt.m+seg.c_seg_cnt) ){ /* center */
        abs_m = seg.spm + seg.t_sz.m1 + seg.t_sz.m2;
        abs_m += ((m-seg.t_cnt.m)/seg.c_cnt.m)*nb;
        if( (m-seg.t_cnt.m) % seg.c_cnt.m ){
            abs_m += seg.c_sz.m1;
        }
    }else{ /* bottom edge */
        abs_m = seg.mpm - (seg.b_sz.m1 + seg.b_sz.m2);
        if( m - (seg.t_cnt.m+seg.c_seg_cnt) ){
            abs_m += seg.b_sz.m1;
        }
    }

    if( right ){
        abs_n += seg.mpn-seg.spn;
    }
    if( bottom ){
        abs_m += seg.mpm-seg.spm;
    }

    *m_tile = abs_m/mb;
    *n_tile = abs_n/nb;
    *offset = (abs_m%mb)*nb+(abs_n%nb);

    return;
}

int type_index_to_sizes(seg_info_t seg, int mb, int nb, int type_index, int *m_off, int *n_off, int *m_sz, int *n_sz){
    int width, height;
    int abs_m, abs_n;
    int type_index_n, type_index_m;
    int success = 1;

    type_index_n = type_index%6;
    type_index_m = type_index/6;

    switch(type_index_n){
        /**** left edge ****/
        case 0:
            abs_n = seg.spn;
            width = seg.l_sz.n1;
            break;
        case 1:
            abs_n = seg.spn;
            width = seg.l_sz.n1;
            abs_n += seg.l_sz.n1;
            width = seg.l_sz.n2;
            break;
        /**** center ****/
        case 2:
            abs_n = seg.spn + seg.l_sz.n1 + seg.l_sz.n2;
            width = seg.c_sz.n1;
            break;
        case 3:
            abs_n = seg.spn + seg.l_sz.n1 + seg.l_sz.n2;
            abs_n += seg.c_sz.n1;
            width = seg.c_sz.n2;
            break;
        /**** right edge ****/
        case 4:
            abs_n = seg.mpn - (seg.r_sz.n1 + seg.r_sz.n2);
            width = seg.r_sz.n1;
            break;
        case 5:
            abs_n = seg.mpn - (seg.r_sz.n1 + seg.r_sz.n2);
            abs_n += seg.r_sz.n1;
            width = seg.r_sz.n2;
            break;
        default: assert(0);
    }

    switch(type_index_m){
        /**** top edge ****/
        case 0:
            abs_m = seg.spm;
            height = seg.t_sz.m1;
            break;
        case 1:
            abs_m = seg.spm;
            abs_m += seg.t_sz.m1;
            height = seg.t_sz.m2;
            break;
        /**** center ****/
        case 2:
            abs_m = seg.spm + seg.t_sz.m1 + seg.t_sz.m2;
            height = seg.c_sz.m1;
            break;
        case 3:
            abs_m = seg.spm + seg.t_sz.m1 + seg.t_sz.m2;
            abs_m += seg.c_sz.m1;
            height = seg.c_sz.m2;
            break;
        /**** bottom edge ****/
        case 4:
            abs_m = seg.mpm - (seg.b_sz.m1 + seg.b_sz.m2);
            height = seg.b_sz.m1;
            break;
        case 5:
            abs_m = seg.mpm - (seg.b_sz.m1 + seg.b_sz.m2);
            abs_m += seg.b_sz.m1;
            height = seg.b_sz.m2;
            break;
    }

    if( !height || !width ){
        success = 0;
    }

    *m_off = abs_m%mb;
    *n_off = abs_n%nb;
    *m_sz = height;
    *n_sz = width;

    return success;
}

int segment_to_arena_index(dague_seg_ddesc_t but_ddesc, int m, int n){
    /* if using named types in the JDF or the default type, then you need to
     * offset the following value by the number of named+default types used
     */
    return segment_to_type_index(but_ddesc.seg_info, m, n);
}

static int segment_to_type_index(seg_info_t seg, int m, int n){
    int type_index_n, type_index_m, type_index;

    if( n >= seg.tot_seg_cnt_n || m >= seg.tot_seg_cnt_m ){
        fprintf(stderr,"invalid segment coordinates\n");
        return -1;
    }

    if( n >= seg.tot_seg_cnt_n/2 ){
        n -= seg.tot_seg_cnt_n/2;
    }
    if( m >= seg.tot_seg_cnt_n/2 ){
        m -= seg.tot_seg_cnt_n/2;
    }

    /* Horizontal */
    if( n < seg.l_cnt.n ){ /* left edge */
        type_index_n = 0;
        if( 1 == n ){
            type_index_n = 1;
        }
    }else if( n < (seg.l_cnt.n+seg.c_seg_cnt) ){ /* center */
        type_index_n = 2;
        if( (n-seg.l_cnt.n) % seg.c_cnt.n ){
            type_index_n = 3;
        }
    }else{ /* right edge */
        type_index_n = 4;
        if( n - (seg.l_cnt.n+seg.c_seg_cnt) ){
            type_index_n = 5;
        }
    }

    /* Vertical */
    if( m < seg.t_cnt.m ){ /* top edge */
        type_index_m = 0;
        if( 1 == m ){
            type_index_m = 1;
        }
    }else if( m < (seg.t_cnt.m+seg.c_seg_cnt) ){ /* center */
        type_index_m = 2;
        if( (m-seg.t_cnt.m) % seg.c_cnt.m ){
            type_index_m = 3;
        }
    }else{ /* bottom edge */
        type_index_m = 4;
        if( m - (seg.t_cnt.m+seg.c_seg_cnt) ){
            type_index_m = 5;
        }
    }

    type_index = type_index_m*6+type_index_n;

    return type_index;
}

