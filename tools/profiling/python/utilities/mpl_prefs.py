#!/usr/bin/env

sched_colors = {'AP': '#ee4466', 'GD': '#eeaa44', 'LFQ':'#5588cc', 'LTQ':'#33dd60', 'PBQ':'#881188'}
GEMM_names = {'dgetrf': 'gemm', 'dpotrf': 'GEMM'}
kernel_names = {'dgetrf': ['gemm', 'swpback', 'swptrsm', 'getrf', 'bcastback'],
                'dpotrf': ['GEMM', 'SYRK', 'TRSM', 'POTRF'],
                'dgeqrf': ['ztsmqr','ztsqrt','dormqr','zgeqrt',]
}
