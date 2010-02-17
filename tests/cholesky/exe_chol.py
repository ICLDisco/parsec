# -*- coding: utf-8 -*-

import sys
import commands

EXE = ["./dposv_ll"]

def get_nb(exe):
  n = 500
  cores = 1
  cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 %s %d %d %d 1 %d" % (exe, cores, n, n, n)
  st, out = commands.getstatusoutput(cmd)
  perf_line = find_perf_line(out)
  nb = int(perf_line.split()[4])
  return nb

def find_perf_line(txt):
  s = "PLASMA DPOTRF"
  for l in txt.split("\n"):
    if l.find(s) >= 0:
      return l
  raise ValueError, "Cannot find '%s' in this:\n%s" % (s, txt)

def run(n, exe):
  for cores in [1, 2, 4, 8]:
    cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  %s %d %d %d 1 %d" % (exe, cores, n, n, n)
    print "####", cmd
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
    for n in range(nb, 50 * nb + 1, nb):
      run(n, exe)

  return 0

sys.exit(main(sys.argv))
