#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import math
import cPickle
from parsec_trials import ParsecTrial, ParsecTrialSet
import subprocess

####### global parameter defaults for ig
# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
default_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IB_divs = {'dgeqrf': 8, 'dpotrf': 0, 'dgetrf': 0}
####### global parameter defaults

min_N = 12288
max_N = 12288
default_exes = ['dpotrf', 'dgeqrf', 'dgetrf' ]
default_scheds = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
default_NBs = [192, 256, 380]
N_hi_mult = 20

def generate_trials(out_dir, print_only=True, Ns=None, min_N=min_N, max_N=max_N,
                    NBs=default_NBs, exes=default_exes, scheds=default_scheds,
                    min_cores=0, max_cores=0, IB_divs=[0],
                    extra_args = []):
    IB_divs_orig = IB_divs

    import socket
    hostname = socket.gethostname().split('.')[0]
    trial_sets = []

    if not Ns:
        generated_Ns = True
    else:
        generated_Ns = False
    # first, generate intended trial sets:
    for ex in exes:
        if not NBs:
            NBs = [default_NBs[ex]]
        for NB in NBs:
            if generated_Ns:
                if NB >= 256:
                    fact = NB
                else:
                    fact = 2 * NB
                Ns = list(range(NB*8 * N_hi_mult, min_N, -fact*8))
                while Ns[0] >= max_N: # cutoff
                    Ns = Ns[1:]
            for N in Ns:
                sys.stderr.write("%s %d\n" % (ex.upper(), N))
                if not IB_divs_orig:
                    IB_divs = [default_IB_divs[ex]]
                elif 'potrf' in ex or 'getrf' in ex:
                    IB_divs = [0] # because they don't use IB
                else:
                    IB_divs = IB_divs_orig
                for IB_div in IB_divs:
                    if IB_div > 0:
                        if NB % IB_div == 0:
                            IB = NB / IB_div
                        else:
                            continue # this one's not a fair combo
                    else:
                        IB = 0
                    for cores in range(min_cores, max_cores + 1):
                        if cores == 0 and os.path.exists('/proc/cpuinfo'):
                            temp = subprocess.check_output(['grep',  '-c', '^processor',
                                                            '/proc/cpuinfo'])
                            try:
                                temp_int = int(temp)
                                cores = temp_int
                            except:
                                cores = 0 # didn't work, so back to default
                        for scheduler in scheds:
                            if not os.path.isdir(out_dir):
                                os.mkdir(out_dir)
                            trial_set = ParsecTrialSet(hostname, ex, N, cores,
                                                       NB, IB, scheduler, extra_args)
                            print(trial_set.shared_name() + ' ' + str(extra_args))
                            if not print_only:
                                file_ = open(out_dir + os.sep + 'pending.' +
                                             trial_set.shared_name(), 'w')
                                trial_set.pickle(file_)
                                file_.close()
                            trial_sets.append(trial_set)
    return trial_sets

def smart_parse(arg, conv=int):
    if not instanceof(arg, str):
        if len(arg) > 1:
            return arg # already an interesting list
        else:
            arg = arg[0] #

    if ',' in arg:
        lst = [conv(x) for x in arg.split(',')]
    elif ':' in arg:
        splits = arg.split(':')
        if len(splits) == 2:
            start, stop = splits
            step = 1
        elif len(splits) == 3:
            start, stop, step = splits
        lst = xrange(int(start), int(stop), int(step))
    else:
        lst = [conv(arg)]
    return lst

###########
## MAIN
###########
if __name__ == '__main__':
    print_only = False
    extra_args = []
    out_dir = '.'

    Ns = None
    NBs = default_NBs
    exes = default_exes
    scheds = default_scheds

    try:
        for arg in sys.argv[1:]:
            if arg == '-l':
                print_only = True
            elif os.path.isdir(arg):
                out_dir = arg
            elif arg.startswith('-N='):
                arg = arg.replace('-N=', '')
                Ns = smart_parse(arg)
            elif arg.startswith('-NB='):
                arg = arg.replace('-NB=', '')
                NBs = smart_parse(arg)
            elif arg.startswith('-exe='):
                exes = smart_parse(arg.replace('-exe=', ''), conv=str)
            elif arg.startswith('-sched='):
                scheds = smart_parse(arg.replace('-sched=', ''), conv=str)
            else:
                extra_args.append(arg)
    except:
        print('Usage: generate_trials.py [OUTPUT_DIRECTORY]' +
              ' -[l]' +
              ' [EXTRA ARGUMENTS TO TEST EXECUTABLE]')
        sys.exit(-1)

    trial_sets = generate_trial_sets(out_dir, print_only=print_only, Ns=Ns, NBs=NBs,
                                     exes=exes, scheds=scheds,
                                     extra_args = extra_args)
    print('generated', len(trial_sets), 'trial sets.')
