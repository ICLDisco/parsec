#!/usr/bin/env

sched_colors = {
    'ap': '#ee4466', # red
    'gd': '#eeaa44', # orange
    'ip': 'brown',
    'lfq':'#5588cc', # blue
    'ltq':'#33dd60', # green
    'pbq':'#881188', # purple
    'rnd':'black',
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

