#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import commands

EXE = ["./dist_exec"]

def get_nb(exe):
  n = 500
  np = 1
  cores = 1
  cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/mpirun -x LD_LIBRARY_PATH -host n1,n2,n3,n4,n5,n6,n7,n8 -bynode -nolocal -np %d %s -c %d -n %d" % (np, exe, cores, n )
  st, out = commands.getstatusoutput(cmd)
  perf_line = find_perf_line(out)
  nb = int(perf_line.split()[4])
  return nb

def find_perf_line(txt):
  s = "GFLOPS"
  for l in txt.split("\n"):
    if l.find(s) >= 0:
      return l
  raise ValueError, "Cannot find '%s' in this:\n%s" % (s, txt)

def run(n, exe):
  for np in [1, 2, 3, 4]:
    for cores in [1, 2, 3, 4]:
      cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -mca btl mx,self -x LD_LIBRARY_PATH -host n5,n6,n7,n8 -bynode -nolocal -np %d  %s -c %d -n %d" % (np, exe, cores, n)
      print "####", cmd
      for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        st, out = commands.getstatusoutput(cmd)
        print find_perf_line(out)
        sys.stdout.flush()

def test_exe(fname):
  try:
    f = open(fname)
  except:
    print "Need executable called %s" % fname
    raise

def main(argv):
  for exe in (EXE):
      test_exe(exe)

  nb = get_nb(EXE[0])
  for exe in (EXE):
    for n in range(7500, 7600, 500):
      run(n, exe)

  return 0

sys.exit(main(sys.argv))
