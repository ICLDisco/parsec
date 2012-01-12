typedef struct{
    int ip;
    int jp;
    unsigned long w;
    unsigned long h;
    unsigned long offset;
}btile_info_t;


btile_info_t map_B_tile_to_A(int i, int j){
    int T1w, T2w, T3w, T4w, T5w, T6w, T7w, T8w, T9w;
    int T1h, T2h, T3h, T4h, T5h, T6h, T7h, T8h, T9h;
    int full_tiles_half_A_h, full_tiles_half_A_w, full_tiles_half_B_h, full_tiles_half_B_w;
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
    full_tiles_half_A_h = (N/2)/mb;
    full_tiles_half_A_w = (N/2)/nb;
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
        }else{
          /* Else take the height of T1 */
          Tc.h = T1h;
        }
    }else if( (i == full_tiles_half_B_h) && (T5h > 0) ){ /* if we are at the middle cleanup tile */
        Tc.ip = full_tiles_half_A_h;

        Tc.h = T5h;
    }else if( (i == 2*full_tiles_half_B_h+1) && (T5h > 0) ){ /* if we are at the bottom cleanup tile */
        Tc.ip = (N/mb);

        Tc.h = T5h;
    }else{ /* if we are in the bottom half, but not at a cleanup tile */
        int i2;

        if( T5h > 0 ){
            /* If the matrix has cleanup tiles (N%mb)!=0 */
            i2 = i-1; /* ignore the cleanup tile of the top half */
        }else{
            /* If the matrix has _no_ cleanup tiles (N%mb)==0 */
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
            /* there is a T4 (i.e. (N/2)%nb!=0), then every odd indexed */
            /* B-tile of the bottom half will be T4 */
            if( (i2%2)==1 && (T4h>0) ){
                Tc.h = T4h;
            }else{
                Tc.h = T1h;
            }
        }
    }


    /* given argument "j" compute "jp" (really Tc.jp) such that B(..., j) belongs to A(..., jp) */
    /* Also derive the width, Tc.w, of the B-tile B(..., j) */

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
        }else{
          /* Else take the width of T1 */
          Tc.w = T1w;
        }
    }else if( (j == full_tiles_half_B_w) && (T7w > 0) ){ /* if we are at the middle cleanup tile */
        Tc.jp = full_tiles_half_A_w;

        Tc.w = T7w;
    }else if( (j == 2*full_tiles_half_B_w+1) && (T7w > 0) ){ /* if we are at the right-most cleanup tile */
        Tc.jp = (N/nb);

        Tc.w = T7w;
    }else{ /* if we are in the right half, but not at a cleanup tile */
        int j2;

        if( T7w > 0 ){
            /* If the matrix has cleanup tiles (N%nb)!=0 */
            j2 = j-1; /* ignore the cleanup tile of the left half */
        }else{
            /* If the matrix has _no_ cleanup tiles (N%nb)==0 */
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
            /* there is a T4 (i.e. (N/2)%nb!=0), then every odd indexed */
            /* B-tile of the right half will be T4 */
            if( (j2%2)==1 && (T4w>0) ){
                Tc.w = T4w;
            }else{
                Tc.w = T1w;
            }
        }
    }


    return Tc;
}


int rank_of(int i, int j){
    btile_info_t Tc;

    Tc = map_B_tile_to_A(i, j);

    return ddescA.super.super.rank_of(Tc.ip, Tc.jp);
}


PLASMA_Complex64_t* data_of(int i, int j){
    uintptr_t ptr;
    btile_info_t Tc;

    Tc = map_B_tile_to_A(i, j);

    ptr  = (uintptr_t)ddescA.super.super.data_of(Tc.ip, Tc.jp);
    ptr += Tc.offset*sizeof(PLASMA_Complex64_t);
    return (PLASMA_Complex64_t*)ptr;
}
