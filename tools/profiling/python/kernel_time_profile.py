#!/usr/bin/python

import sys, os
import subprocess
import re
from shutil import move, copy
import profile2dat
import process_profile
from random import randint
import math

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
        self.Gstddev = 0.0
        self.avgTime = 0.0
        self.Tstddev = 0.0
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
        return ("%s %s %d cores %d NB %d IB %d gflops(avg/stddev): %f %f time(avg/stddev): %f %f" %
                (self.name, self.sched, self.matrixSize,
                 self.numCores, self.NB, self.IB, self.avgGflops, self.Gstddev, self.avgTime, self.Tstddev))

if __name__ == '__main__':
   matrixSizes = [3600, 9000, 14400, 21600, 27000, 31500]
   minNumCores = 0 # default to using them all
   maxNumCores = 0
   NBs = [180] # use default
   IBdivs = [1]
   numTrials = 10
   names = [ 'dpotrf']
   testingDir = 'dplasma/testing/'
   outputBaseDir = '/mnt/scratch/pgaultne/' # = testingDir
   schedulers = ['LTQ', 'LFQ']

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
                  for scheduler in schedulers:
                     schedDir = outputBaseDir + scheduler + os.sep
                     if not os.path.isdir(schedDir):
                        os.mkdir(schedDir)
                     set = TrialSet(name, matrixSize, numCores, NB, IB, scheduler)
                     outputDir = schedDir + str(matrixSize) + os.sep
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
                             marker = randint(0, 99999)
                             print("AN ERROR OCCURRED %d" % marker)
                             sys.stderr.write(str(marker) + ':\n' + stderr + '\n')
                         match = pattern.match(stdout)
                         if match:
                             set.NB = int(match.group(2))
                             gflops = float(match.group(3))
                             time = float(match.group(1))
                             print("gflops: %f time: %f NB:%d" % (gflops, time, set.NB))
                             trialObj = Trial(gflops, time)
                             set.append(trialObj)
                         else:
                             sys.stderr.write("results not properly parsed: %s\n" % stdout)
                         # move profiling file to unique name
                         profile2dat.profiles2dat(testingDir, 
                                                  outputDirectory = outputDir,
                                                  filePrefix = 'testing_' + set.name, 
                                                  outputTag = 't' + str(trialNum),
                                                  unlink = True)
                     gflopsSet = []
                     timeSet = []
                     for trial in set:
                         timeSet.append(trial.time)
                         gflopsSet.append(trial.gflops)
                     set.Gstddev, set.avgGflops = process_profile.online_variance_mean(gflopsSet)
                     set.Tstddev, set.avgTime = process_profile.online_variance_mean(timeSet)
                     set.Gstddev = math.sqrt(set.Gstddev)
                     set.Tstddev = math.sqrt(set.Tstddev)
                     trialSets.append(set)
                     print set # early progress report

   print '\n\nfinal results:\n'
   for trialSet in trialSets:
       print trialSet
   print 'done'
