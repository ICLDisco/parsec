#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import math
import cPickle
from parsec_trials import Trial, TrialSet
import subprocess

####### global parameter defaults for ig            
# it would be nice to have different 'default experiment parameters'
# for different machines (e.g. ig, zoot).
#
# even nicer would be a more configurable way of generating trial sets.
# so that, for instance, I could specify different NB and IB params depending on
# the scheduler
default_NBs = {'dgeqrf': 192, 'dpotrf': 256, 'dgetrf': 256 }
default_IBdivs = {'dgeqrf': 8, 'dpotrf': 0, 'dgetrf': 0}
####### global parameter defaults

def generate_trial_sets(output_base_dir, list_only = True, extra_args = []):
    #
    # customize this section to your heart's content!
    #
    # defaults for ig:
    execs = ['dpotrf'] #, 'dgetrf_incpiv'] #, 'dpotrf', 'dgeqrf' ]
    schedulers = ['AP', 'GD', 'LTQ', 'LFQ', 'PBQ']
    minNumCores = 0 # default to using them all
    maxNumCores = 0
    minN = 7000
    maxN = 21000
    NBs = [160, 188, 200, 216, 256]
    IBdivs = None    # use defaults
    Ns = None        # generated based on tile size
    N_hi_mult = 20
    #
    # overrides
    #
#    IBdivs = [1,2,4,8,11] # None to use default per exec

#    IBdivs = [2, 4]
    NBs = [180, 380, 400]        # None to use defaults

#    Ns = [15360]
#    NBs = [180, 200, 360, 380]
    NBs = [256]
#    IBdivs = [1,2,8]
    IBdivs = [0]
    #
    # end customizable param section
    #

    #
    # transfer dict items
    #
    
    IBdivs_orig = IBdivs
    
    import socket
    hostname = socket.gethostname().split('.')[0]
    trial_sets = []

    if not Ns:
        generated_Ns = True
    else:
        generated_Ns = False
    # first, generate intended trial sets:
    for ex in execs:
        if not NBs:
            NBs = [default_NBs[ex]]
        for NB in NBs:
            if generated_Ns:
                if NB >= 256:
                    fact = NB
                else:
                    fact = 2 * NB
                Ns = list(range(NB*8 * N_hi_mult, minN, -fact*8))
                while Ns[0] >= maxN: # cutoff
                    Ns = Ns[1:]
            for N in Ns:
                sys.stderr.write("%s %d\n" % (ex.upper(), N))   
                if not IBdivs_orig:
                    IBdivs = [default_IBdivs[ex]]
                elif 'potrf' in ex or 'getrf' in ex:
                    IBdivs = [0] # because they don't use IB
                else:
                    IBdivs = IBdivs_orig
                for IBdiv in IBdivs:
                    if IBdiv > 0:
                        if NB % IBdiv == 0:
                            IB = NB / IBdiv
                        else:
                            continue # this one's not a fair combo
                    else:
                        IB = 0
                    for cores in range(minNumCores,maxNumCores + 1):
                        if cores == 0 and os.path.exists('/proc/cpuinfo'):
                            temp = subprocess.check_output(['grep',  '-c', '^processor',
                                                            '/proc/cpuinfo'])
                            try:
                                temp_int = int(temp)
                                cores = temp_int
                            except:
                                cores = 0 # didn't work, so back to default
                        for scheduler in schedulers:
                            if not os.path.isdir(output_base_dir):
                                os.mkdir(output_base_dir)
                            trial_set = TrialSet(hostname, ex, N, cores,
                                                 NB, IB, scheduler, extra_args)
                            print(trial_set.shared_name() + ' ' + str(extra_args))
                            if not list_only:
                                # save planned file in case everything dies
                                trial_set.pickle(output_base_dir + os.sep + 'pending.' +
                                                 trial_set.shared_name())
                            trial_sets.append(trial_set)
    return trial_sets
    
###########
## MAIN
###########
if __name__ == '__main__':
    list_only = False
    extra_args = []
    output_base_dir = '.'
    
    try:
        for arg in sys.argv[1:]:
            if arg == '-l':
                list_only = True
            elif os.path.isdir(arg):
                output_base_dir = arg
            else:
                extra_args.append(arg)
    except:
        print('Usage: generate_tests.py [OUTPUT_DIRECTORY]' +
              ' -[l]' +
              ' [EXTRA ARGUMENTS TO TEST EXECUTABLE]')
        sys.exit(-1)

    generate_trial_sets(output_base_dir, list_only = list_only,
                        extra_args = extra_args)
