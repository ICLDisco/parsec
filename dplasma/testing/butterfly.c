#include <stdint.h>

typedef struct{
    int ip;
    int jp;
    int w;
    int h;
    uintptr_t offset;
}btile_info_t;


btile_info_t map_B_tile_to_A(int i, int j, int Np, int mb, int nb){
    int T1w, T2w, T3w, T4w, T5w, T6w, T7w, T8w, T9w;
    int T1h, T2h, T3h, T4h, T5h, T6h, T7h, T8h, T9h;
    int full_tiles_half_A_h, full_tiles_half_A_w, full_tiles_half_B_h, full_tiles_half_B_w;
    int off_w, off_h;
    btile_info_t Tc;

    T4w = (Np/2)%nb;
    T4h = (Np/2)%mb;
    T1w = nb - T4w;
    T1h = mb - T4h;
    T2w = T1w;
    T2h = T4h;
    T3w = T4w;
    T3h = T1h;

    T5w = T1w;
    if( T4h < T1h ){
        T5h = T4h;
    }else if( T4h == T1h ){
        T5h = 0;
    }else if( T4h > T1h ){
        T5h = T4h-T1h;
    }

    T6w = T3w;
    T6h = T5h;

    T7w = T5h;
    T7h = T5w;
    T8w = T6h;
    T8h = T6w;

    T9w = T7w;
    T9h = T5h;


    /* See how many whole tiles of A fit in the top half and the left half of  */
    /* the L0 butterfly and calculate how many full tiles of B fit in the same */
    /* top/left half.  Due to symmetry, top and left are the same for L0, but  */
    /* not for L1.                                                             */
    full_tiles_half_A_h = (Np/2)/mb;
    full_tiles_half_A_w = (Np/2)/nb;
    if( T4h == 0 ){
        full_tiles_half_B_h = full_tiles_half_A_h;
    }else{
        full_tiles_half_B_h = 2*full_tiles_half_A_h;
    }
    if( T4h >= T1h ){
        /* In this case the last A tile (which does not fit in the top half) */
        /* has a T1 B-tile that fits whole in the top half */
        full_tiles_half_B_h++;
    }

    if( T7w == 0 ){
        full_tiles_half_B_w = full_tiles_half_A_w;
    }else{
        full_tiles_half_B_w = 2*full_tiles_half_A_w;
    }
    if( T7w >= T1w ){
        /* In this case the last A tile (which does not fit in the left half) */
        /* has a T1 B-tile that fits whole in the left half */
        full_tiles_half_B_w++;
    }

    /* given argument "i" compute "ip" (really Tc.ip) such that B(i,...) belongs to A(ip,...) */
    /* Also derive the height, Tc.h, of the B-tile B(i,...) */
    /* And the offset height, off_h, of the B-tile inside the A-tile */

    if( i < full_tiles_half_B_h ){ /* If we are in the top half */
        if( T4h > 0 ){
            /* If there are two B-tiles per A tile */
            Tc.ip = i/2;
        }else{
            Tc.ip = i;
        }

        if( (i%2)==1 && T4h>0 ){
          /* If my row says I'm the second B-tile in a A-tile and there */
          /* are two B-tiles per A-tile, take the height of T4 */
          Tc.h = T4h;
          off_h = T1h;
        }else{
          /* Else take the height of T1 */
          Tc.h = T1h;
          off_h = 0;
        }
    }else if( (i == full_tiles_half_B_h) && (T5h > 0) ){ /* if we are at the middle cleanup tile */
        Tc.ip = full_tiles_half_A_h;
        Tc.h = T5h;
        if( (i%2)==1 ){ /* If there is a T5, there _has_ to be a T4 (and a T2) */
            off_h = T1h;
        }else{
            off_h = 0;
        }
    }else if( (i == 2*full_tiles_half_B_h+1) && (T5h > 0) ){ /* if we are at the bottom cleanup tile */
        Tc.ip = (Np/mb);
        Tc.h = T5h;
        if( (full_tiles_half_B_h%2)==1 ){
            /* If the middle cleanup B-tile was in the middle of an A-tile, */
            /* then the bottom cleanup B-tile is at the beginning of an A-tile */
            off_h = 0;
        }else{
            /* If the middle cleanup B-tile was at the beginning of an A-tile, */
            /* then the bottom cleanup B-tile is after a "bottom" B-tile (T2/T4) */
            off_h = T4h;
        }
    }else{ /* if we are in the bottom half, but not at a cleanup tile */
        int i2, i3;

        if( T5h > 0 ){
            /* If the matrix has cleanup tiles (Np%mb)!=0 */
            i2 = i-1; /* ignore the cleanup tile of the top half */
        }else{
            /* If the matrix has _no_ cleanup tiles (Np%mb)==0 */
            i2 = i; /* there is no cleanup tile to ignore */
        }

        if( T4h > 0 ){
            /* If there are two B-tiles per A tile */
            Tc.ip = i2/2;
        }else{
            Tc.ip = i2;
        }

        /* See if the first B-tile of the bottom half has an even or odd index */
        if( full_tiles_half_B_h > 2*full_tiles_half_A_h ){ /* which means that T4h >= T1h (and also that T4h > 0) */
            /* If there was an odd number of B-tiles on the top half, then */
            /* every odd indexed B-tile of the bottom half will be T1 */
            if( (i2%2)==1 ){
                Tc.h = T1h;
            }else{
                Tc.h = T4h;
            }
        }else{
            /* If there was an even number of B-tiles on the top half _and_ */
            /* there is a T4 (i.e. (Np/2)%nb!=0), then every odd indexed */
            /* B-tile of the bottom half will be T4 */
            if( (i2%2)==1 && (T4h>0) ){
                Tc.h = T4h;
            }else{
                Tc.h = T1h;
            }
        }

        i3 = i2-full_tiles_half_B_h;
        if( (i3%2) == 0 ){
            off_h = (Np/2)%mb;
        }else{
            off_h = 0;
        }
    }


    /* given argument "j" compute "jp" (really Tc.jp) such that B(..., j) belongs to A(..., jp) */
    /* Also derive the width, Tc.w, of the B-tile B(..., j) */
    /* And the offset width, off_w, of the B-tile inside the A-tile */

    if( j < full_tiles_half_B_w ){ /* If we are in the left half */
        if( T4w > 0 ){
            /* If there are two B-tiles per A tile */
            Tc.jp = j/2;
        }else{
            Tc.jp = j;
        }

        if( (j%2)==1 && T4w>0 ){
          /* If my column says I'm the second B-tile in a A-tile and there */
          /* are two B-tiles per A-tile, take the width of T4 */
          Tc.w = T4w;
          off_w = T1w;
        }else{
          /* Else take the width of T1 */
          Tc.w = T1w;
          off_w = 0;
        }
    }else if( (j == full_tiles_half_B_w) && (T7w > 0) ){ /* if we are at the middle cleanup tile */
        Tc.jp = full_tiles_half_A_w;
        Tc.w = T7w;
        if( (j%2)==1 ){ /* If there is a T7, there _has_ to be a T4 (and a T3) */
            off_w = T1w;
        }else{
            off_w = 0;
        }
    }else if( (j == 2*full_tiles_half_B_w+1) && (T7w > 0) ){ /* if we are at the right-most cleanup tile */
        Tc.jp = (Np/nb);
        Tc.w = T7w;
        if( (full_tiles_half_B_w%2)==1 ){
            /* If the middle cleanup B-tile was in the middle of an A-tile, */
            /* then the right-most cleanup B-tile is at the beginning of an A-tile */
            off_w = 0;
        }else{
            /* If the middle cleanup B-tile was at the beginning of an A-tile, */
            /* then the right-most cleanup B-tile is after a "right" B-tile (T3/T4) */
            off_w = T4w;
        }
    }else{ /* if we are in the right half, but not at a cleanup tile */
        int j2, j3;

        if( T7w > 0 ){
            /* If the matrix has cleanup tiles (Np%nb)!=0 */
            j2 = j-1; /* ignore the cleanup tile of the left half */
        }else{
            /* If the matrix has _no_ cleanup tiles (Np%nb)==0 */
            j2 = j; /* there is no cleanup tile to ignore */
        }
    
        if( T4w > 0 ){
            /* If there are two B-tiles per A tile */
            Tc.jp = j2/2;
        }else{
            Tc.jp = j2;
        }

        /* See if the first B-tile of the right half has an even or odd index */
        if( full_tiles_half_B_w > 2*full_tiles_half_A_w ){ /* which means that T4w >= T1w (and also that T4w > 0) */
            /* If there was an odd number of B-tiles on the left half, then */
            /* every odd indexed B-tile of the right half will be T1 */
            if( (j2%2)==1 ){
                Tc.w = T1w;
            }else{
            /* Otherwise, every odd indexed B-tile of the right half will be T4 */
                Tc.w = T4w;
            }
        }else{
            /* If there was an even number of B-tiles on the left half _and_ */
            /* there is a T4 (i.e. (Np/2)%nb!=0), then every odd indexed */
            /* B-tile of the right half will be T4 */
            if( (j2%2)==1 && (T4w>0) ){
                Tc.w = T4w;
            }else{
                Tc.w = T1w;
            }
        }

        j3 = j2-full_tiles_half_B_w;
        if( (j3%2) == 0 ){
            off_w = (Np/2)%nb;
        }else{
            off_w = 0;
        }
    }

    Tc.offset = (uintptr_t)(off_h*mb+off_w);

    return Tc;
}


int rank_of(int i, int j){
    btile_info_t Tc;

    Tc = map_B_tile_to_A(i, j, Np, mb, nb);

    return dcA.super.super.rank_of(Tc.ip, Tc.jp);
}


PLASMA_Complex64_t* data_of(int i, int j){
    uintptr_t ptr=0;
    btile_info_t Tc;

    Tc = map_B_tile_to_A(i, j, Np, mb, nb);

    ptr  = (uintptr_t)dcA.super.super.data_of(Tc.ip, Tc.jp);
    ptr += Tc.offset*sizeof(PLASMA_Complex64_t);
    return (PLASMA_Complex64_t*)ptr;
}
