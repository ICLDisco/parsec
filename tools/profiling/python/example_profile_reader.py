#! /usr/bin/env python

import dbpreader_py as pread
import sys

if __name__ == '__main__':
    filenames = ['testing_dpotrf.profile']
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    profile = pread.readProfile(filenames)
    print('done reading!')
    info = None
    for fl in profile.files:
        for thr in fl.threads:
            for ev in thr.events:
                # if ev.handle_id == 0 and ev.key == profile.dictionary['EXEC_MISSES'].id:
                # 	print('oh, a different handle')
                if ev.handle_id == 1: # and ev.key == profile.dictionary['EXEC_MISSES'].id:
                    print ev
                if ev.key == profile.dictionary['EXEC_MISSES'].id:
                    if not ev.info:
                        print('that sucks bigtime')
                    info = ev.info
    print(profile.files[0].threads[0].events[0].row_header())
    print(profile.dictionary['EXEC_MISSES'].id)
    print(profile.handle_counts)
    print(profile.thread_count)
    print(info.row_header())
    print(info)

