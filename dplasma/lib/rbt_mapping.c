#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "rbt_mapping.h"


seg_info_t calculate_constants(input_t args){
    //int spy, spx, mpy, mpx, epy, epx;
    int ay, ax, by, bx, cy, cx, dy, dx, ey, ex, fy, fx;
    int nb,mb,N,l,ib,jb;
    int xy, width, height, w;
    seg_info_t seg;

    memset(&seg, 0, sizeof(seg_info_t));

    N  = args.N;
    nb = args.nb;
    l  = args.L;
    ib = args.ib;
    jb = args.jb;

    mb = nb;
    w = 1<<l;

    assert(ib>=jb);

    /* Calculate starting, middle and ending point for this butterfly */
    seg.spy = ib*N/w;
    seg.spx = jb*N/w;

    seg.epy = (ib+1)*N/w-1;
    seg.epx = (jb+1)*N/w-1;

    seg.mpy = (seg.spy + seg.epy + 1)/2;
    seg.mpx = (seg.spx + seg.epx + 1)/2;

    /* Calculate the different sizes that might appear */
    ay = seg.spy%nb;
    by = mb-ay;
    cy = seg.mpy%nb;
    dy = mb-cy;
    ey = (dy>by) ? dy-by : nb + (dy-by);
    fy = mb-ey;

    ax = seg.spx%nb;
    bx = mb-ax;
    /* FIXME: If we only have two original tiles in a butterfly quarter, it can be cx := ex */
    cx = seg.mpx%nb;
    dx = mb-cx;
    ex = (dx>bx) ? dx-bx : mb + (dx-bx);
    fx = mb-ex;

    /* top center edge types */
    if( by < fy+ey ) {
        if( fy < by ) {
            height = by-fy;
            seg.t_cnt.y = 2;
            seg.t_sz.y1 = height;
            seg.t_sz.y2 = fy;
        }else{
            height = by;
            seg.t_cnt.y = 1;
            seg.t_sz.y1 = height;
            seg.t_sz.y2 = 0;
        }
    }

    /* left center edge types */
    if( bx < fx+ex ) {
        if( fx < bx ) {
            width = bx-fx;
            seg.l_cnt.x = 2;
            seg.l_sz.x1 = width;
            seg.l_sz.x2 = fx;
        }else{
            width = bx;
            seg.l_cnt.x = 1;
            seg.l_sz.x1 = width;
            seg.l_sz.x2 = 0;
        }
    }

    /* right edge types */
    if( cx < ex+fx && cx ) {
        if( ex < cx ){
            width = cx-ex;
            seg.r_cnt.x = 2;
            seg.r_sz.x1 = ex;
            seg.r_sz.x2 = width;
        }else{
            width = cx;
            seg.r_cnt.x = 1;
            seg.r_sz.x1 = width;
            seg.r_sz.x2 = 0;
        }
    }

    /* bottom edge types */
    if( cy < ey+fy && cy) {
        if( ey < cy ){
            height = cy-ey;
            seg.b_cnt.y = 2;
            seg.b_sz.y1 = ey;
            seg.b_sz.y2 = height;
        }else{
            height = cy;
            seg.b_cnt.y = 1;
            seg.b_sz.y1 = height;
            seg.b_sz.y2 = 0;
        }
    }

    /* center types */
    do{
        int startx, endx;

        if( 0 < fx ){
            seg.c_cnt.x = 2;
            seg.c_cnt.y = 2;
        }else{
            seg.c_cnt.x = 1;
            seg.c_cnt.y = 1;
        }
        seg.c_sz.x1 = ex;
        seg.c_sz.x2 = fx;
        seg.c_sz.y1 = ey;
        seg.c_sz.y2 = fy;

        startx = seg.spx;
        if( bx != nb ){
          startx += bx;
        }
        endx = seg.mpx;
        if( cx != nb ){
            endx -= cx;
        }

        seg.c_seg_cnt = seg.c_cnt.x*(endx-startx)/nb;
    }while(0); // just to give me a scope without looking ugly.

    seg.tot_seg_cnt_x = 2*(seg.l_cnt.x + seg.c_seg_cnt + seg.r_cnt.x);
    seg.tot_seg_cnt_y = 2*(seg.t_cnt.y + seg.c_seg_cnt + seg.b_cnt.y);

    return seg;
}


void find_tile(seg_info_t seg, int mb, int nb, int i, int j){
    int width, height;
    int abs_i, abs_j;
    int right=0, bottom=0;
    int quart_seg_x, quart_seg_y;

    quart_seg_x = (seg.l_cnt.x+seg.c_seg_cnt+seg.r_cnt.x);
    quart_seg_y = (seg.t_cnt.y+seg.c_seg_cnt+seg.b_cnt.y);

    if( j >= 2*quart_seg_x || i >= 2*quart_seg_y ){
        fprintf(stderr,"invalid segment coordinates\n");
        return;
    }

    if( j >= quart_seg_x ){
        j -= quart_seg_x;
        right = 1;
    }
    if( i >= quart_seg_y ){
        i -= quart_seg_y;
        bottom = 1;
    }

    /* Horizontal */
    if( j < seg.l_cnt.x ){ /* left edge */
        abs_j = seg.spx;
        width = seg.l_sz.x1;
        if( 1 == j ){
            abs_j += seg.l_sz.x1;
            width = seg.l_sz.x2;
        }
    }else if( j < (seg.l_cnt.x+seg.c_seg_cnt) ){ /* center */
        abs_j = seg.spx + seg.l_sz.x1 + seg.l_sz.x2;
        abs_j += ((j-seg.l_cnt.x)/seg.c_cnt.x)*nb;
        width = seg.c_sz.x1;
        if( (j-seg.l_cnt.x) % seg.c_cnt.x ){
            abs_j += seg.c_sz.x1;
            width = seg.c_sz.x2;
        }
    }else{ /* right edge */
        abs_j = seg.mpx - (seg.r_sz.x1 + seg.r_sz.x2);
        width = seg.r_sz.x1;
        if( j - (seg.l_cnt.x+seg.c_seg_cnt) ){
            abs_j += seg.r_sz.x1;
            width = seg.r_sz.x2;
        }
    }

    /* Vertical */
    if( i < seg.t_cnt.y ){ /* top edge */
        abs_i = seg.spy;
        height = seg.t_sz.y1;
        if( 1 == i ){
            abs_i += seg.t_sz.y1;
            height = seg.t_sz.y2;
        }
    }else if( i < (seg.t_cnt.y+seg.c_seg_cnt) ){ /* center */
        abs_i = seg.spy + seg.t_sz.y1 + seg.t_sz.y2;
        abs_i += ((i-seg.t_cnt.y)/seg.c_cnt.y)*nb;
        height = seg.c_sz.y1;
        if( (i-seg.t_cnt.y) % seg.c_cnt.y ){
            abs_i += seg.c_sz.y1;
            height = seg.c_sz.y2;
        }
    }else{ /* bottom edge */
        abs_i = seg.mpy - (seg.b_sz.y1 + seg.b_sz.y2);
        height = seg.b_sz.y1;
        if( i - (seg.t_cnt.y+seg.c_seg_cnt) ){
            abs_i += seg.b_sz.y1;
            height = seg.b_sz.y2;
        }
    }

    if( right ){
        abs_j += seg.mpx-seg.spx;
    }
    if( bottom ){
        abs_i += seg.mpy-seg.spy;
    }
    printf("(%d,%d), off: %d, HxW: %dx%d\n",abs_i/mb, abs_j/nb, (abs_i%mb)*nb+(abs_j%nb), height,width);

    return;
}
