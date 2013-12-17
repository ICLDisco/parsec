#!/usr/bin/env

sched_colors = {
    'AP': '#ee4466',
    'GD': '#eeaa44',
    'IP': 'brown',
    'LFQ':'#5588cc',
    'LTQ':'#33dd60',
    'PBQ':'#881188',
    'RND':'black',
}

GEMM_names = {
    'dgetrf': 'gemm',
    'dpotrf': 'GEMM',
    'dgeqrf': 'ztsmqr'
}

kernel_names = {
    'dgetrf': ['gemm', 'swpback', 'swptrsm', 'getrf', 'bcastback'],
    'dpotrf': ['GEMM', 'SYRK', 'TRSM', 'POTRF'],
    'dgeqrf': ['ztsmqr','ztsqrt','dormqr','zgeqrt',]
}

task_slice_colors = {
    'Execution': 'seagreen',
    'Selection': 'darkblue',
    'Starvation': 'lightskyblue',
    'Preparation': 'maroon',
    'Completion': 'gold',
    'Framework': 'gray',
}

colors = {
    'sched': sched_colors,
}
