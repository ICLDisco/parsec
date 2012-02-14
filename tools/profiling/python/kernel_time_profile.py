#!/usr/bin/python

import sys, os
import subprocess
import re
from shutil import move, copy

class Trial(object):
    def __init__(self, gflops, runtime):
        self.gflops = gflops
        self.time = runtime
        
class TrialSet(list):
    def __init__(self, name, matrixSize, numCores=0, NB=0, IB=0, sched='LFQ'):
        self.matrixSize = matrixSize
        self.numCores = numCores
        self.NB = NB
        self.IB = IB
        self.name = name
        self.avgGflops = 0.0
        self.stddev = 0.0
        self.avgTime = 0.0
        self.sched = sched
    def genCmd(self):
        cmd = 'testing_' + self.name
        args = []
        args.append('-N')
        args.append(str(self.matrixSize))
        args.append('-o')
        args.append(self.sched)
        if self.numCores > 0:
            args.append('-c')
            args.append(str(self.numCores))
        if self.NB > 0:
            args.append('-NB')
            args.append(str(self.NB))
        if self.IB > 0:
            args.append('-IB')
            args.append(str(self.IB))
        return cmd, args
    def __str__(self):
        return ("set name %s, size %d, cores %d, NB %d, IB %d, average gflops: %f, avg time: %f" %
                (self.name, self.matrixSize,
                 self.numCores, self.NB, self.IB, self.avgGflops, self.avgTime))
        
if __name__ == '__main__':  
   matrixSizes = [3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600]
   minNumCores = 0 # default to using them all
   maxNumCores = 0
   NBs = [0] # use default
   IBdivs = [1]
   numTrials = 10
   names = [ 'dpotrf']
   testingDir = 'dplasma/testing/'
   scheduler = 'LTQ'

   pattern = re.compile("### TIMED\s(\d+\.\d+)\s+s.+?NB=\s+(\d+).+?(\d+\.\d+)\s+gflops$")

   trialSets = []
   # per actual testing script
   for name in names:
      for matrixSize in matrixSizes:
         sys.stderr.write("%s %d\n" % (name, matrixSize))
         for numCores in range(minNumCores,maxNumCores + 1):
            for NB in NBs:
                for IBdiv in IBdivs:
                    IB = NB / IBdiv
                    set = TrialSet(name, matrixSize, numCores, NB, IB, scheduler)
                    outputDir = testingDir + os.sep + str(matrixSize) + os.sep
                    if not os.path.isdir(outputDir):
                        os.mkdir(outputDir)
                    for trialNum in range(0, numTrials):
                        print("%s for %dx%d matrix on %d cores, NB = %d, IB = %d; sched = %s trial #%d" %
                              (name, matrixSize, matrixSize, numCores, NB, IB, set.sched, trialNum))
                        cmd, args = set.genCmd()
                        proc = subprocess.Popen([testingDir + cmd] + args,
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        (stdout, stderr) = proc.communicate()
                        if len(stderr) != 0:
                            print("AN ERROR OCCURRED")
                            print(stderr)
                        else:
                            match = pattern.match(stdout)
                            if match:
                                set.NB = int(match.group(2))
                                gflops = float(match.group(3))
                                time = float(match.group(1))
                                print("gflops: %f time: %f NB:%d" % (gflops, time, set.NB))
                                trialObj = Trial(gflops, time)
                                set.append(trialObj)
                            else:
                                print("results not properly parsed: %s" % stdout)
                            # move profiling file to unique name
                            move(testingDir + 'testing_' + set.name + '.profile',
                                 outputDir + set.name + '.' + str(set.matrixSize) + '.' +
                                 str(trialNum) + '.profile')
                    totalGflops = 0.0
                    totalTime = 0.0
                    for item in set:
                        totalGflops += item.gflops
                        totalTime += item.time
                    set.avgGflops = totalGflops / len(set)
                    set.avgTime = totalTime / len(set)
                    trialSets.append(set)
                    print set # early progress report

   for trialSet in trialSets:
       print trialSet
   print 'done'
