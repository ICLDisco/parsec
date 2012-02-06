#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void usage(char *name){
    fprintf(stderr,"Usage: %s  -N matrix-size  -nb tile-size  -L level\n", name);
    exit(-1);
}

void parse_args(int *N_ptr, int *nb_ptr, int *L_ptr, char **argv, int argc){

    if( argc < 5 ){
        usage(argv[0]);
    }
    while(argc>1){
        if( !strcmp(argv[argc-2], "-N") ){
            *N_ptr = atoi(argv[argc-1]);
        }else if( !strcmp(argv[argc-2], "-nb") ){
            *nb_ptr = atoi(argv[argc-1]);
        }else if( !strcmp(argv[argc-2], "-L") ){
            *L_ptr = atoi(argv[argc-1]);
        }else{
            usage(argv[0]);
        }
        argc -= 2;
    }

    return;
}


int main(int argc, char **argv){
    int i,j,nb,mb,N,L,l;

    parse_args(&N, &nb, &L, argv, argc);
    mb = nb;

    for(l=0; l<=L; l++){
        int w = 1<<l;
        for(i=0;i<w;i++){
            for(j=0;j<w;j++){
                int ax,bx,cx,dx,ex,fx;
                int ay,by,cy,dy,ey,fy;
                /* (starting/ending/middle) (position/tile) (x/y) */
                int spx, spy, epx, epy, mpx, mpy, stx, sty, etx, ety, mtx, mty;
                int type_count, type_count_x, lt_cnt_x, org_t, type_count_y, lt_cnt_y;

                printf("------------------------\n");

                spy = i*N/w;
                spx = j*N/w;
                sty = spy/mb;
                stx = spx/nb;

                epy = (i+1)*N/w-1;
                epx = (j+1)*N/w-1;
                ety = epy/mb;
                etx = epx/nb;

                mpy = (spy+epy+1)/2;
                mpx = (spx+epx+1)/2;
                mty = mpy/mb;
                mtx = mpx/nb;

                ay = spy%nb;
                by = mb-ay;
                cy = mpy%nb;
                dy = mb-cy;
                ey = (dy>by) ? dy-by : nb + (dy-by);
                fy = mb-ey;

                ax = spx%nb;
                bx = mb-ax;
/* FIXME: If we only have two original tiles in a butterfly quarter, it can be cx := ex */
                cx = mpx%nb;
                dx = mb-cx;
                ex = (dx>bx) ? dx-bx : mb + (dx-bx);
                fx = mb-ex;

                /***** Horizontal *****/

                /* original tiles in a horizontal sweep of the */
                /* butterfly's top left quarter                */
                org_t = mtx-stx+1;
                /* subtract the edges */
                org_t -= 2;
                /* bound it */
                if( org_t < 0 )
                    org_t = 0;

                printf("org_t: %d, mtx: %d, stx: %d\n",org_t, mtx, stx);
                /* count the types inside the "central", repeating tile */
                type_count_x = 1;
                if( fx > 0 ){
                    type_count_x++;
                }

                /* start calculating the horizontal local tile count */
                lt_cnt_x = org_t*type_count_x;

                /* add one, or two, more tiles (due to the right edge) to the */
                /* local count and add one more type, if needed */
                if( cx > ex ){
                    lt_cnt_x+=2;
                }else if( cx > 0 ){
                    lt_cnt_x++;
                }
                if( (cx != (fx+ex)) && (cx != ex) ){
                    type_count_x++;
                }

                /* add one, or two, more tiles (due to the left edge) to the */
                /* local count and add one more type, if needed */
                if( (fx == 0) || (bx < fx) ){
                    lt_cnt_x++;
                }else if( bx > fx ){
                    lt_cnt_x+=2;
                }
                if( (bx != (fx+ex)) && (bx != fx) ){
                    type_count_x++;
                }


                /***** Vertical *****/

                /* original tiles in a vertical sweep of the   */
                /* butterfly's top left quarter                */
                org_t = mty-sty+1;
                /* subtract the edges */
                org_t -= 2;
                /* bound it */
                if( org_t < 0 )
                    org_t = 0;

                /* count the types inside the "central", repeating tile */
                type_count_y = 1;
                if( fy > 0 ){
                    type_count_y++;
                }

                /* start calculating the vertical local tile count */
                lt_cnt_y = org_t*type_count_y;

                /* add one, or two, more tiles (due to the right edge) to the */
                /* local count and add one more type, if needed */
                if( cy > ey ){
                    lt_cnt_y+=2;
                }else if( cy > 0 ){
                    lt_cnt_y++;
                }
                if( (cy != (fy+ey)) && (cy != ey) ){
                    type_count_y++;
                }

                /* add one, or two, more tiles (due to the left edge) to the */
                /* local count and add one more type, if needed */
                if( (fy == 0) || (by < fy) ){
                    lt_cnt_y++;
                }else if( by > fy ){
                    lt_cnt_y+=2;
                }
                if( (by != (fy+ey)) && (by != fy) ){
                    type_count_y++;
                }

                /************************/

                type_count = type_count_x * type_count_y;

                printf("L         : %d\n",l);
                printf("i,j       : %d,%d\n",i,j);
                printf("Type count: %d\n",type_count);
                printf("ax, bx    : %d, %d\n",ax, bx);
                printf("ex, fx    : %d, %d\n",ex, fx);
                printf("cx, dx    : %d, %d\n",cx, dx);
                printf("ay, by    : %d, %d\n",ay, by);
                printf("ey, fy    : %d, %d\n",ey, fy);
                printf("cy, dy    : %d, %d\n",cy, dy);
                printf("local mt  : %d\n",lt_cnt_y);
                printf("local nt  : %d\n",lt_cnt_x);


                /* left center edge types */
                if( bx < fx+ex && bx != fx ) {
                    int width;
                    if( fx < bx ) {
                        width = bx-fx;
                    }else{
                        width = bx;
                    }

                    printf("Type LC   : %dx%d\n",ey,width);
                    if(fy > 0){
                        printf("Type LC2  : %dx%d\n",fy,width);
                    }
                }

                /* central types */
                printf("Type CM   : %dx%d\n",ey,ex);
                if( fx > 0 ){
                    printf("Type CM2  : %dx%d\n",fy,fx);
                    printf("Type CM3  : %dx%d\n",fy,ex);
                    printf("Type CM4  : %dx%d\n",ey,fx);
                }

                /* right edge types */
                if( cx < ex+fx && cx != ex ) {
                    int width;
                    if( ex < cx ){
                        width = cx-ex;
                    }else{
                        width = cx;
                    }

                    printf("Type RC   : %dx%d\n",ey,width);
                    if( fy > 0 )
                        printf("Type RC2  : %dx%d\n",fy,width);
                }


                /* top center edge types */
                if( by < fy+ey && by != fy ) {
                    int height;
                    if( fy < by ) {
                        height = by-fy;
                    }else{
                        height = by;
                    }

                    printf("Type TC   : %dx%d\n",height, ex);
                    if(fx > 0){
                        printf("Type TC2  : %dx%d\n",height, fx);
                    }
                }

                /* bottom edge types */
                if( cy < ey+fy && cy != ey ) {
                    int height;
                    if( ey < cy ){
                        height = cy-ey;
                    }else{
                        height = cy;
                    }

                    printf("Type BC   : %dx%d\n",height, ex);
                    if( fx > 0 )
                        printf("Type BC2  : %dx%d\n",height, fx);
                }

                /* top left corner types */
                if( ax && (bx < fx+ex) && ay && (by < fy+ey) && (bx != ex || by != ey) ) {
                    int width, height;
                    if( fx < bx ) {
                        width = bx-fx;
                    }else{
                        width = bx;
                    }
                    if( fy < by ) {
                        height = by-fy;
                    }else{
                        height = by;
                    }

                    printf("Type TL   : %dx%d\n",height,width);
                    if( (0 < fy) && (fy < by) && (fx != bx)){
                        printf("Type TL2  : %dx%d\n",fy,width);
                    }
                    if( (0 < fx) && (fx < bx) && (fy != by)){
                        printf("Type TL3  : %dx%d\n",height,fx);
                    }
                }

                /* bottom left corner types */
                if( ax && (bx < fx+ex) && dy && (cy < fy+ey) && (bx != ex || cy != ey) ) {
                    int width, height;
                    if( fx < bx ) {
                        width = bx-fx;
                    }else{
                        width = bx;
                    }
                    if( ey < cy ) {
                        height = cy-ey;
                    }else{
                        height = cy;
                    }

                    printf("Type BL   : %dx%d\n",height,width);
                    if( (0 < fy) && (ey < cy) && (fx != bx) ){
                        printf("Type BL2  : %dx%d\n",ey,width);
                    }
                    if( (0 < fx) && (fx < bx) && (ey != cy) ){
                        printf("Type BL3  : %dx%d\n",height,fx);
                        /*
                         * This is NOT a new type, it's central type CM4
                        if( (0 < fy) && (ey < cy) ){
                            printf("Type BL4  : %dx%d\n",ey,fx);
                        }
                        */
                    }
                }

                /* top right corner types */
                if( dx && (cx < fx+ex) && ay && (by < fy+ey) && (cx != ex || by != ey) ) {
                    int width, height;
                    if( ex < cx ) {
                        width = cx-ex;
                    }else{
                        width = cx;
                    }
                    if( fy < by ) {
                        height = by-fy;
                    }else{
                        height = by;
                    }

                    printf("Type TR   : %dx%d\n",height,width);
                    if( (0 < fy) && (fy < by) && (ex != cx)){
                        printf("Type TR2  : %dx%d\n",fy, width);
                    }
                    if( (0 < fx) && (fx < cx) && (fy != by)){
                        printf("Type TR3  : %dx%d\n",height, ex);
                    }
                }

                /* bottom right corner types */
                if( dx && (cx < fx+ex) && dy && (cy < fy+ey) && (cx != ex || cy != ey) ) {
                    int width, height;
                    if( ex < cx ) {
                        width = cx-ex;
                    }else{
                        width = cx;
                    }
                    if( ey < cy ) {
                        height = cy-ey;
                    }else{
                        height = cy;
                    }

                    printf("Type BR   : %dx%d\n",height,width);
                    if( (0 < fy) && (ey < cy) && (fx != cx) ){
                        printf("Type BR2  : %dx%d\n",ey,width);
                    }
                    if( (0 < fx) && (fx < bx) && (ey != cy) ){
                        printf("Type BR3  : %dx%d\n",height,ex);
                    }
                }

            }
        }
    }
    return 0;
}
