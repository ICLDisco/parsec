#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import commands

EXE = ["./sposv_rl"]
repeat = 10
tile_size = 256
gpu_array = [0,1,2]
core_array = [1, 2, 4, 8]

def get_nb(exe):
  n = 256
  cores = 1
  gpu = 1
  cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 %s -c %d --gpu %d %d" % (exe, cores, gpu, n)
  print "####", cmd
  st, out = commands.getstatusoutput(cmd)
  perf_line = find_block_size(out)
  nb = int(perf_line.split()[3])
  return nb

def find_block_size(txt):
  s = "Dplasma initialization:"
  for l in txt.split("\n"):
    if l.find(s) >= 0:
      return l
  raise ValueError, "Cannot find '%s' in this:\n%s" % (s, txt)

def find_perf_line(txt):
  s = "Dplasma computation:"
  for l in txt.split("\n"):
    if l.find(s) >= 0:
      return l
  return "Cannot find '%s' in this:\n%s" % (s, txt)

def run(file, n, exe):
  for cores in core_array:
    for gpu in gpu_array:
      cmd = "env GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  %s -c %d --gpu %d -B %d %d" % (exe, cores, gpu, tile_size, n)
      print "####", cmd
      for count in range(0, repeat, 1):
        st, out = commands.getstatusoutput(cmd)
        perf_output = str(cores) + " x " + str(gpu) + " x " + str(n) + " " + find_perf_line(out)
        print perf_output
        temp = "\n#\n# " + perf_output + "\n#\n"
        file.write(temp)
        file.write(out)
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

  nb = tile_size
  for exe in (EXE):
    file = open(exe + "_" + str(tile_size) + ".dat", 'w')
    for n in range(nb, 50 * nb + 1, nb):
      run(file, n, exe)
    file.close()

  return 0

sys.exit(main(sys.argv))
