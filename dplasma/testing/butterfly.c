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

    /* See how many whole tiles of A fit in the top half of the L0 butterfly */
    /* and calculate how many full tiles of B fit in the same top half */
    full_tiles_half_A = (N/2)/mb;
    if( T4h == 0 ){
        full_tiles_half_B = full_tiles_half_A;
    }else{
        full_tiles_half_B = 2*full_tiles_half_A;
    }
    if( T4h >= T1h ){
        full_tiles_half_B++;
    }

    /* given argument "i" compute "ip" such that B(i,...) belongs to A(ip,...) */
    /* Also derive the height of B-tile B(i,...) */

    if( i < full_tiles_half_B ){ /* If we are in the top half */
        if( T4h>0 ){
            Tc.ip = i/2;
        }else{
            Tc.ip = i;
        }

        if( (i%2)==1 && T4h>0 ){
          /* If my column says I'm the second B-tile in a A-tile and there */
          /* are two B-tiles per A-tile, take the height of T4 */
          Tc.h = T4h;
        }else{
          /* Else take the height of T1 */
          Tc.h = T1h;
        }
    }else if( (i == full_tiles_half_B) && (T5h > 0) ){ /* if we are at the middle cleanup tile */
        Tc.ip = full_tiles_half_A;

        Tc.h = T5h;
    }else if( (i == 2*full_tiles_half_B+1) && (T5h > 0) ){ /* if we are at the bottom cleanup tile */
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
            Tc.ip = i2/2;
        }else{
            Tc.ip = i2;
        }

        /* See if the first B-tile of the bottom half has an even or odd index */
        if( full_tiles_half_B > 2*full_tiles_half_A ){ /* which means that T4h >= T1h (and also that T4h > 0) */
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

#error "work in progress"

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
