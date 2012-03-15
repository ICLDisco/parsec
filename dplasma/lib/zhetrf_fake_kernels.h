#ifndef _ZHETRF_FAKE_KERNELS_H_
#define _ZHETRF_FAKE_KERNELS_H_

void CORE_ztrmdm(int uplo, int tempkn, void *tile, int ldak );

void CORE_zgemdm(int trans, int conjtrans, int tempmm, 
    int tempnn, int tempkn, PLASMA_Complex64_t mzone, 
    void *tile0, int ld1, void *tile1, 
    int mb, PLASMA_Complex64_t zone, void *tile2,
    int ld2, void *tile3, int ld, void *pool_ptr, int lwork);

void CORE_zhedrk(int uplo, int trans, int tempmm, 
    int tempkn, int ib, int aaa, 
    void *tile0, int ld1, int bbb,
    void *tile1, int ld2, void *tile2,
    int ld, void *pool_ptr, int lwork );

void CORE_zhetrf2_nopiv(int uplo, int tempkn, int ib, 
    void *tile0, int ldak, void *pool_ptr,
    int ldwork, int ld );

void CORE_zhetrf_nopiv(int uplo, int tempkn, int ib, 
    void *tile0, int ldak, void *pool_ptr,
    int ldwork, int ld );


void CORE_ztrmdm(int uplo, int tempkn, void *tile, int ldak ){
    (void)uplo;
    (void)tempkn;
    (void)tile;
    (void)ldak;
}

void CORE_zgemdm(int trans, int conjtrans, int tempmm, 
	int tempnn, int tempkn, PLASMA_Complex64_t mzone, 
	void *tile0, int ld1, void *tile1, 
	int mb, PLASMA_Complex64_t zone, void *tile2,
	int ld2, void *tile3, int ld, void *pool_ptr, int lwork){

    (void) trans;
    (void) conjtrans;
    (void) tempmm;
	(void) tempnn;
    (void) tempkn;
    (void) mzone;
    (void) tile0;
    (void) ld1;
    (void) tile1;
    (void) mb;
    (void) zone;
    (void) tile2;
	(void) ld2;
    (void) tile3;
    (void) ld;
    (void) pool_ptr;
    (void) lwork;
}

void CORE_zhedrk(int uplo, int trans, int tempmm, 
	int tempkn, int ib, int aaa, 
	void *tile0, int ld1, int bbb,
	void *tile1, int ld2, void *tile2,
	int ld, void *pool_ptr, int lwork ){

    (void)uplo;
    (void)trans;
    (void)tempmm;
    (void)tempkn;
    (void)ib;
    (void)aaa;
    (void)tile0;
    (void)ld1;
    (void)bbb;
    (void)tile1;
    (void)ld2;
    (void)tile2;
    (void)ld;
    (void)pool_ptr;
    (void)lwork;

}


void CORE_zhetrf_nopiv(int uplo, int tempkn, int ib, 
	void *tile0, int ldak, void *pool_ptr,
	int ldwork, int ld ){

    (void)uplo;
    (void)tempkn;
    (void)ib;
    (void)tile0;
    (void)ldak;
    (void)pool_ptr;
    (void)ldwork;
    (void)ld;

}


void CORE_zhetrf2_nopiv(int uplo, int tempkn, int ib, 
	void *tile0, int ldak, void *pool_ptr,
	int ldwork, int ld ){

    (void)uplo;
    (void)tempkn;
    (void)ib;
    (void)tile0;
    (void)ldak;
    (void)pool_ptr;
    (void)ldwork;
    (void)ld;
}

#endif
