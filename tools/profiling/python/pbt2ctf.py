#!/usr/bin/env python3

import h5toctf
import sys
import os

try:
    import pbt2ptt
except ModuleNotFoundError:
    print("Did not find pbt2ptt, you are likely using python version that does not match the version used to build PaRSEC profiling tools")
    print(sys.path)

def bool(str):
    return str.lower() in ["true", "yes", "y", "1", "t"]

def pbt_to_ctf(pbt_files_list, ctf_filename, skip_parsec_events, skip_mpi_events):
    print(f"Converting {pbt_files_list} into a HDF5 File")
    ptt_filename = pbt2ptt.convert(pbt_files_list, multiprocess=False)
    h5toctf.h5_to_ctf(ptt_filename, ctf_filename, skip_parsec_events, skip_mpi_events)

if __name__ == "__main__":

    pbt_file_prefix = sys.argv[1]
    ctf_file_name = sys.argv[2]
    skip_parsec_events = True
    skip_mpi_events = True
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        sys.exit("usage: pbt_to_ctf.py <pbt base filename> <ctf filename> [skip PaRSEC events? default=1] [skip MPI events? default=1]")
    if len(sys.argv) >= 4:
        skip_parsec_events = bool(sys.argv[3])
    if len(sys.argv) >= 5:
        skip_mpi_events = bool(sys.argv[4])

    # iterate over all files within the directory that start with sys.argv[1]
    pbt_files_list=[]
    dirname = os.path.dirname(pbt_file_prefix)
    for file in os.listdir(dirname):
        file_fullname = os.path.join(dirname,file)
        if file_fullname.startswith(pbt_file_prefix) and file_fullname.endswith(".prof") and file_fullname != ctf_file_name:
            print("found file ", file_fullname)
            pbt_files_list.append(file_fullname)

    # to debug: read_pbt(pbt_files_list[0]), etc.
    pbt_to_ctf(pbt_files_list, ctf_file_name, skip_parsec_events, skip_mpi_events)
