#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/mca/device/level_zero/device_level_zero.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/runtime.h"
#include "parsec/execution_stream.h"

#include <oneapi/mkl.hpp>
#include <sys/time.h>

extern "C" {
    extern int gemm_lz_verbose;
    int gemm_kernel_lz(parsec_device_gpu_module_t *gpu_device,
                     parsec_gpu_task_t *gpu_task,
                     parsec_gpu_exec_stream_t *gpu_stream);
}

int gemm_kernel_lz(parsec_device_gpu_module_t *gpu_device,
                   parsec_gpu_task_t *gpu_task,
                   parsec_gpu_exec_stream_t *gpu_stream)
{
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    parsec_task_t *this_task = gpu_task->ec;
    struct timeval start, end, diff;
    double delta;
    double *a_gpu, *b_gpu, *c_gpu;
    parsec_level_zero_exec_stream_t *lz_stream = (parsec_level_zero_exec_stream_t *)gpu_stream;

    (void)gpu_device;

    parsec_dtd_unpack_args(this_task,
                           &A, &B, &C,
                           &m, &n, &k,
                           &mb, &nb, &kb);

    a_gpu = static_cast<double*>(parsec_dtd_get_dev_ptr(this_task, 0));
    b_gpu = static_cast<double*>(parsec_dtd_get_dev_ptr(this_task, 1));
    c_gpu = static_cast<double*>(parsec_dtd_get_dev_ptr(this_task, 2));

    gettimeofday(&start, NULL);

    double alpha=0.0;
    double beta=1.0;
    try {
      oneapi::mkl::blas::gemm(lz_stream->swq->queue, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
         mb, kb, nb,
         alpha, static_cast<const double*>(a_gpu), mb,
         static_cast<const double*>(b_gpu), kb,
         beta, static_cast<double*>(c_gpu), nb);
    } catch (const oneapi::mkl::invalid_argument &e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws invalid argument exception");
    } catch (const oneapi::mkl::unsupported_device &e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws unsuported device exception");
    } catch (const oneapi::mkl::host_bad_alloc &e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws host bad allocation exception");
    } catch (const oneapi::mkl::device_bad_alloc &e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws device bad allocation exception");
    } catch (const oneapi::mkl::unimplemented &e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws unimplemented exception");
    } catch (const std::exception& e) {
      parsec_warning("OneAPI MKL BLAS GEMM throws unexpected exception");
    } catch (...) {
      parsec_warning("OneAPI MKL BLAS GEMM throws unexpected exception that is also badly formatted...");
    }
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
    if(gemm_lz_verbose)
        fprintf(stderr, "GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, GPU %s submitted in %g s\n",
                m, n, k, mb, kb, kb, nb, mb, kb,
                this_task->taskpool->context->my_rank,
                gpu_stream->name, delta);

    return PARSEC_HOOK_RETURN_DONE;
}
