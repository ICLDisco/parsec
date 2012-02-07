#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static int sf=12; /* scale factor */
int spx, mpx;

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

void rect(int x, int y, int w, int h, const char *c, int s){
    if( s > 0 ){
        fprintf(stderr,"  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%d\"/>\n",sf*x, sf*y, sf*w, sf*h, c, s);
    }else{
        int off = mpx-spx;
        fprintf(stderr,"  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke-width=\"0\"/>\n",sf*x, sf*y, sf*w, sf*h, c);
        fprintf(stderr,"  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke-width=\"0\"/>\n",sf*(x+off), sf*y, sf*w, sf*h, c);
        fprintf(stderr,"  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke-width=\"0\"/>\n",sf*x, sf*(y+off), sf*w, sf*h, c);
        fprintf(stderr,"  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke-width=\"0\"/>\n",sf*(x+off), sf*(y+off), sf*w, sf*h, c);
    }
    return;
}

void line(int x1, int y1, int x2, int y2, char *c, int w){
    fprintf(stderr,"  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%d\" />\n",sf*x1,sf*y1,sf*x2,sf*y2,c,w);
    return;
}

int main(int argc, char **argv){
    int i,j,nb,mb,N,L,l;
    int xy;

    parse_args(&N, &nb, &L, argv, argc);
    mb = nb;

fprintf(stderr,"<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\"\n",sf*N,sf*N,sf*N,sf*N);
fprintf(stderr,"     xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n");

    l = L;
    int w = 1<<l;
    for(i=0;i<w;i++){
        for(j=0;j<w;j++){
            int ax,bx,cx,dx,ex,fx;
            int ay,by,cy,dy,ey,fy;
            /* (starting/ending/middle) (position/tile) (x/y) */
            int spy, epx, epy, mpy, stx, sty, etx, ety, mtx, mty;
            int type_count, type_count_x, lt_cnt_x, org_t, type_count_y, lt_cnt_y;

            if( j > i ) continue;

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


            type_count = 0;

            /* top left corner types */
            if( ax && (bx < fx+ex) && ay && (by < fy+ey) ) {
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
                rect(spx+0, spy+0, width, height, "#ffff00", 0);
                if( (0 < fy) && (fy < by) && (fx != bx)){
                    printf("Type TL2  : %dx%d\n",fy,width);
                    rect(spx+0, spy+height, width, fy, "#ff0000", 0);
                }
                if( (0 < fx) && (fx < bx) && (fy != by)){
                    printf("Type TL3  : %dx%d\n",height,fx);
                    rect(spx+width, spy+0, height, fx, "#700020", 0);
                    if( (0 < fy) && (fy < by) ){
                        printf("Type TL4  : %dx%d\n",fy,fx);
                        rect(spx+width, spy+height, fy, fx, "#c0c0ff", 0);
                    }
                }
            }

            /* bottom left corner types */
            if( ax && (bx < fx+ex) && dy && (cy < fy+ey) ) {
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
                rect(spx+0, mpy-cy+((cy>ey)?ey:0), width, height, "#006000", 0);
                if( (0 < fy) && (ey < cy) && (fx != bx) ){
                    printf("Type BL2  : %dx%d\n",ey,width);
                    rect(spx+0, mpy-cy, width, ey, "#20b000", 0);
                }
                if( (0 < fx) && (fx < bx) ){
                    printf("Type BL3  : %dx%d\n",height,fx);
                    rect(spx+width, mpy-cy+((cy>ey)?ey:0), fx, height, "#5000a0", 0);
                    if( (0 < fy) && (ey < cy) ){
                        printf("Type BL4  : %dx%d\n",ey,fx);
                        rect(spx+width, mpy-cy, ey, fx, "#e0e0e0", 0);
                    }
                }
            }

            /* top right corner types */
            if( dx && (cx < fx+ex) && ay && (by < fy+ey) ) {
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
                rect(mpx-cx+((cx>ex)?ex:0), spy+0, width, height, "#202020", 0);
                if( (0 < fy) && (fy < by) ){
                    printf("Type TR2  : %dx%d\n",fy, width);
                    rect(mpx-cx+((cx>ex)?ex:0), spy+height, width, fy, "#505050", 0);
                }
                if( (0 < fx) && (ex < cx) && (fy != by)){
                    printf("Type TR3  : %dx%d\n",height, ex);
                    rect(mpx-cx, spy+0, ex, height, "#85a590", 0);
                    if( (0 < fy) && (fy < by) ){
                        printf("Type TR4  : %dx%d\n",fy, ex);
                        rect(mpx-cx, spy+by-fy, ex, fy, "#d0d0d0", 0);
                    }
                }
            }

            /* bottom right corner types */
            if( dx && (cx < fx+ex) && dy && (cy < fy+ey) ) {
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
                rect(mpx-cx+((cx>ex)?ex:0), mpy-cy+((cy>ey)?ey:0), width, height, "#b06060", 0);
                if( (0 < fy) && (ey < cy) && (ex != cx) ){
                    printf("Type BR2  : %dx%d\n",ey,width);
                    rect(mpx-cx+((cx>ex)?ex:0), mpy-cy, width, ey, "#60b060", 0);
                }
                if( (0 < fx) && (ex < cx) ){
                    printf("Type BR3  : %dx%d\n",height,ex);
                    rect(mpx-cx, mpy-cy+((cy>ey)?ey:0), ex, height, "#6060b0", 0);
                    if( (0 < fy) && (ey < cy) ){
                        printf("Type BR4  : %dx%d\n",ey,ex);
                        rect(mpx-cx, mpy-cy, ex, ey, "#c0c0c0", 0);
                    }
                }
            }

            /* right edge types */
            if( cx < ex+fx ) {
                int width, pos, start, end;

                if( ex < cx ){
                    width = cx-ex;
                }else{
                    width = cx;
                }

                printf("Type RC   : %dx%d\n",ey,width);
                if( ex < cx ){
                    printf("Type RC3  : %dx%d\n",ey,ex);
                }

                start = spy;
                if( by != mb ){
                  start += by;
                }
                end = mpy-ey;
                if( cy != mb ){
                    end -= cy;
                }
                for(pos = start; pos <= end; pos += mb){
                  rect(mpx-cx+((ex<cx)?ex:0), pos, width, ey, "#a0ffb0", 0);
                  if( ex < cx ){
                      rect(mpx-cx, pos, ex, ey, "#c0c090", 0);
                  }
                }

                if( fy > 0 ){
                    printf("Type RC2  : %dx%d\n",fy,width);
                    if( ex < cx )
                        printf("Type RC4  : %dx%d\n",fy,ex);

                    start = spy+ey;
                    if( by != mb ){
                      start += by;
                    }
                    end = mpy-fy;
                    if( cy != mb ){
                        end -= cy;
                    }
                    for(pos = start; pos <= end; pos += mb){
                      rect(mpx-cx+((ex<cx)?ex:0), pos, width, fy, "#a0a0ff", 0);
                      if( ex < cx )
                          rect(mpx-cx, pos, ex, fy, "#c090c0", 0);
                    }
                }
            }


            /* top center edge types */
            if( by < fy+ey ) {
                int height, pos, start, end;
                if( fy < by ) {
                    height = by-fy;
                }else{
                    height = by;
                }

                printf("Type TC   : %dx%d\n",height, ex);
                if( fy < by ) {
                    printf("Type TC2  : %dx%d\n",fy, ex);
                }

                start = spx;
                if( bx != nb ){
                  start += bx;
                }
                end = mpx-ex;
                if( cx != nb ){
                    end -= cx;
                }
                for(pos = start; pos <= end; pos += nb){
                    rect(pos, spy, ex, height, "#ffa0a0", 0);
                    if( fy < by ) {
                        rect(pos, spy+height, ex, fy, "#fff000", 0);
                    }
                }

                if(fx > 0){
                    printf("Type TC2  : %dx%d\n",height, fx);
                    if( fy < by ) {
                        printf("Type TC4  : %dx%d\n",fy, fx);
                    }

                    start = spx+ex;
                    if( bx != nb ){
                      start += bx;
                    }
                    end = mpx-fx;
                    if( cx != nb ){
                        end -= cx;
                    }
                    for(pos = start; pos <= end; pos += nb){
                        //rect(pos, spy, fx, height, "#00fff0", 0);
                        rect(pos, spy, fx, height, "#009080", 0);
                        if( fy < by ) {
                            rect(pos, spy+height, fx, fy, "#ff00f0", 0);
                        }
                    }
                }
            }

            /* bottom edge types */
            if( cy < ey+fy ) {
                int height, pos, start, end;
                if( ey < cy ){
                    height = cy-ey;
                }else{
                    height = cy;
                }

                printf("Type BC   : %dx%d\n",height, ex);
                if( ey < cy ) {
                    printf("Type BC2  : %dx%d\n",ey, ex);
                }

                start = spx;
                if( bx != nb ){
                  start += bx;
                }
                end = mpx-ex-fx;
                if( cx != nb ){
                    end -= cx;
                }
                for(pos = start; pos <= end; pos += nb){
                    //rect(pos, mpy-cy+((ey<cy)?ey:0), ex, height, "#7cfb00", 0);
                    rect(pos, mpy-cy+((ey<cy)?ey:0), ex, height, "#70fb00", 0);
                    if( ey < cy ) {
                        rect(pos, mpy-cy, ex, ey, "#b09070", 0);
                    }
                }

                if( fx > 0 ){
                    printf("Type BC2  : %dx%d\n",height, fx);
                    if( ey < cy ) {
                        printf("Type BC2  : %dx%d\n",ey, fx);
                    }

                    start = spx+ex;
                    if( bx != nb ){
                      start += bx;
                    }
                    end = mpx-fx;
                    if( cx != nb ){
                        end -= cx;
                    }
                    for(pos = start; pos <= end; pos += nb){
                        rect(pos, mpy-cy+((ey<cy)?ey:0), fx, height, "#000040", 0);
                        if( ey < cy ) {
                            rect(pos, mpy-cy, fx, ey, "#400000", 0);
                        }
                    }
                }
            }

            /* left center edge types */
            if( bx < fx+ex && bx != fx ) {
                int width, pos, start, end;
                if( fx < bx ) {
                    width = bx-fx;
                }else{
                    width = bx;
                }

                printf("Type LC   : %dx%d\n",ey,width);
                if( fx < bx ){
                    printf("Type LC3  : %dx%d\n",ey,fx);
                }

                start = spy;
                if( by != mb ){
                  start += by;
                }
                end = mpy-ey-fy;
                if( cy != mb ){
                    end -= cy;
                }
                for(pos = start; pos <= end; pos += mb){
                  //rect(spx, pos, width, ey, "#507090", 0);
                  rect(spx, pos, width, ey, "#8b0000", 0);
                  if( fx < bx ){
                      rect(spx+width, pos, fx, ey, "#d090d0", 0);
                  }
                }

                if(fy > 0){
                    printf("Type LC2  : %dx%d\n",fy,width);
                    if( fx < bx )
                        printf("Type LC4  : %dx%d\n",fy,fx);

                    start = spy+ey;
                    if( by != mb ){
                      start += by;
                    }
                    end = mpy-fy;
                    if( cy != mb ){
                        end -= cy;
                    }
                    for(pos = start; pos <= end; pos += mb){
                      rect(spx, pos, width, fy, "#404000", 0);
                      if( fx < bx ){
                          rect(spx+width, pos, fx, fy, "#faa052", 0);
                      }
                    }
                }

            }

            {
                int width, posx, startx, endx, posy, starty, endy;
                /* central types */
                printf("Type CM   : %dx%d\n",ey,ex);

                starty = spy;
                if( by != mb ){
                  starty += by;
                }
                endy = mpy-ey-fy;
                if( cy != mb ){
                    endy -= cy;
                }

                startx = spx;
                if( bx != mb ){
                  startx += bx;
                }
                endx = mpx-ex-fx;
                if( cx != nb ){
                    endx -= cx;
                }

                for(posx = startx; posx <= endx; posx += nb){
                    for(posy = starty; posy <= endy; posy += mb){
                        rect(posx, posy, ex, ey, "#00008b", 0);
                        if( 0 < fx ){
                            printf("Type CM2  : %dx%d\n",fy,fx);
                            printf("Type CM3  : %dx%d\n",fy,ex);
                            printf("Type CM4  : %dx%d\n",ey,fx);
                            rect(posx+ex, posy, fx, ey, "#008b00", 0);
                            rect(posx, posy+ey, ex, fy, "#00ffff", 0);
                            rect(posx+ex, posy+ey, fx, fy, "#707070", 0);
                        }
                    }
                }

            }

        }
    }

    for(xy=0;xy<N;xy++){
        line(0, xy, N, xy, "#b0b0b0", 1);
        line(xy, 0, xy, N, "#b0b0b0", 1);
    }
    for(xy=0;xy<N;xy+=nb){
        line(0, xy, N, xy, "#000000", 1);
        line(xy, 0, xy, N, "#000000", 1);
    }

    for(xy=0;xy<N;xy+=N/(2*w)){
        line(0, xy, N, xy, "#808080", 3);
        line(xy, 0, xy, N, "#808080", 3);
    }
    for(xy=0;xy<N;xy+=N/w){
        line(0, xy, N, xy, "#000000", 3);
        line(xy, 0, xy, N, "#000000", 3);
    }

    fprintf(stderr,"</svg>\n");


    return 0;
}
