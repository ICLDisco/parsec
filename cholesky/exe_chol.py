# -*- coding: utf-8 -*-

import sys
import commands

DEXE = "./dposv.dplasma" # exe file for DPLASMA
PEXE = "./dposv.plasma" # exe file for PLASMA
JDF = "cholesky.jdf"

def get_nb():
  n = 500
  cores = 1
  cmd = "env GOTO_NUM_THREADS=1 %s %d %d %d 1 %d < %s" % (PEXE, cores, n, n, n, JDF)
  st, out = commands.getstatusoutput(cmd)
  first_line = out.split("\n")[0]
  nb = int(first_line.split()[4])
  return nb

def find_perf_line(txt):
  for l in txt.split("\n"):
    if l.find("PLASMA DPOTRF") >= 0:
      return l

def run(n, exe):
  cores = 1
  cmd = "env GOTO_NUM_THREADS=1 %s %d %d %d 1 %d < %s" % (exe, cores, n, n, n, JDF)
  print "####", cmd
  st, out = commands.getstatusoutput(cmd)
  print find_perf_line(out)

def main(argv):
  nb = get_nb()
  for exe in (PEXE, DEXE):
    for n in range(nb, 10 * nb + 1, nb):
      run(n, exe)

  return 0

sys.exit(main(sys.argv))
