
struct starpu_codelet cl_spotrf = {
    .cpu_funcs  = {spotrf_cpu, NULL},
    .cuda_funcs = {spotrf_cuda, NULL},
    .nbuffers   = 1    
};

void spotrf_cpu(void *descr[], void *args)
{
    int nb, *info;
    float *T;
    starpu_codelet_unpack_args(&cl_spotrf, &nb, info);
    T = STARPU_MATRIX_GET_PTR(descr[0]);
    CORE_spotrf(PlasmaLower, nb, T, nb, info);
}

void spotrf_cuda()
{
    int cuda_nb, *cuda_info;
    float *cuda_T;
    starpu_codelet_unpack_args(&cl_spotrf, &cuda_nb, cuda_info);
    cuda_T = STARPU_MATRIX_GET_PTR(descr[0]);
    magma_spotrf_gpu('L', cuda_nb, (float*)cuda_T, cuda_nb, cuda_info);
    cudaDeviceSynchronize();
}

struct starpu_codelet cl_strsm = {
    .cpu_funcs = {strsm_cpu, NULL},
    .cuda_funcs = {strsm_cuda, NULL},
    .nbuffers = 2   
};

void strsm_cpu(void *descr[], void *arg)
{
    int nb;
    float *T; 
    float *C;
    starpu_codelet_unpack_args(&cl_strsm, &nb);
    T = STARPU_MATRIX_GET_PTR(descr[0]);
    C = STARPU_MATRIX_GET_PTR(descr[1]);
    CORE_strsm(PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
	       nb, nb, 1.0, T, nb, C, nb);
}

void strsm_cuda(void *descr[], void *arg)
{
    int nb;
    float *T, *C;
    starpu_codelet_unpack_args(&cl_strsm, &nb);
    T = STARPU_MATRIX_GET_PTR(descr[0]);
    C = STARPU_MATRIX_GET_PTR(descr[1]);
    cublasStrsm((char)PlasmaRight, (char)PlasmaLower, 
		 (char)PlasmaTrans, (char)PlasmaNonUnit, nb, nb,
		 (float) 1.0,
		 (float *) T, nb, 
		 (float *) C, nb);
    cudaDeviceSynchronize();
}

struct starpu_codelet cl_sherk = {
    .cpu_funcs = {sherk_cpu, NULL},
    .cuda_funcs = {sherk_cuda, NULL},
    .nbuffers = 2
};

void sherk_cpu(void *descr[], void *arg)
{
    int nb;
    int **zz, *zlli;
    float *A, *T;
    starpu_codelet_unpack_args(&cl_herk, &nb);
    A = STARPU_MATRIX_GET_PTR(descr[0]);
    T = STARPU_MATRIX_GET_PTR(descr[1]);
    CORE_ssyrk(PlasmaLower, PlasmaNoTrans, nb, nb,
	       (float)-1.0, (float*)A, nb, (float) 1.0, (float*)T, nb );   
}
    
void sherk_cuda(void *descr[], void *arg)
{    
    int nb;
    float *A, *T;
    starpu_codelet_unpack_args(&cl_herk, &nb);
    A = STARPU_MATRIX_GET_PTR(descr[0]);
    T = STARPU_MATRIX_GET_PTR(descr[1]);
    cublasSsyrk(PlasmaLower, PlasmaNoTrans, nb, nb,
		(float)-1.0, 
		(float *) (A), nb,
		(float) 1.0,
		(float *) (T), nb);
    cudaDeviceSynchronize();
}

struct starpu_codelet cl_sgemm = {
    .cpu_funcs = {sgemm_cpu, NULL},
    .cuda_funcs = {sgemm_cuda, NULL},
    .nbuffers = 3
};

void sgemm_cpu(void *descr[], void *arg)
{
    int nb; 
    float *A, *B, *C;
    starpu_codelet_unpack_args(&cl_sgemm, &nb);
    A = STARPU_MATRIX_GET_PTR(descr[0]);
    B = STARPU_MATRIX_GET_PTR(descr[1]);
    C = STARPU_MATRIX_GET_PTR(descr[2]);
    CORE_sgemm(PlasmaNoTrans, PlasmaTrans, nb, nb, nb,
	       (float)-1.0, A, nb, B, nb,
	       (float) 1.0, C, nb);
}
    
void sgemm_cuda(void *descr[], void *arg)
{
    int nb;
    float *l[5];
    float *A, *B, *C;
    starpu_codelet_unpack_args(&cl_sgemm, &nb);
    A = STARPU_MATRIX_GET_PTR(descr[0]);
    B = STARPU_MATRIX_GET_PTR(descr[1]);
    C = STARPU_MATRIX_GET_PTR(descr[2]);
    cublasSgemm(PlasmaNoTrans, PlasmaTrans, nb, nb, nb,
		(float)  -1.0,
		(float*) (B), nb,
		(float*) (A), nb, 
		(float)  1.0,
		(float*) (C), nb);
    cudaDeviceSynchronize();
}

	
void cholesky(int NB, int SIZE, PLASMA_enum uplo, int* INFO)
{
    int k, m, n;
    
    for(k = 0; k<SIZE; k++)
    {
	//POTRF
	starpu_insert_task(&cl_spotrf,
			   STARPU_VALUE, &NB, sizeof(int),
			   STARPU_VALUE, INFO, sizeof(int),
			   STARPU_RW, A[k][k], 0);

	for(m = k+1; m<SIZE; m++)
	{
	    //TRSM
	    starpu_insert_task(&cl_strsm,
			       STARPU_VALUE, &NB, sizeof(int),
			       STARPU_R, A[k][k],
			       STARPU_RW, A[m][k], 0);
	}
	for(m = k+1; m<SIZE; m++)
	{
	    //HERK
	    starpu_insert_task(&cl_sherk, 
			       STARPU_VALUE, &NB, sizeof(int),
			       STARPU_R, A[m][k],
			       STARPU_RW, A[m][m], 0);
	    for(n = k+1; n<m; n++)
	    {
		// GEMM
		starpu_insert_task(&cl_sgemm,
				   STARPU_VALUE, &NB, sizeof(int),
				   STARPU_R, A[m][k],
				   STARPU_R, A[n][k],
				   STARPU_RW, A[m][n], 0);
	    }
	}
    }
}
