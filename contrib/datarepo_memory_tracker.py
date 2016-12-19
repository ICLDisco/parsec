import sys
import re

if len(sys.argv) == 1:
    print("Usage: " + sys.argv[0] + " trace_file_1 trace_file_2 ...")
    print("  Reads trace files (output from a PaRSEC run) with a debug level of at least 3, and")
    print("  checks that all data repository allocation is correctly freed.")
    sys.exit(1)
sys.argv.pop(0)

alloc_re = re.compile(r"\[D\^PaRSEC *([0-9]+)\]:.*entry (0x[0-9a-f]+/[0-9]+) of.*has been allocated.*")
notfree_re = re.compile(r"\[D\^PaRSEC *([0-9]+)\]:.*entry (0x[0-9a-f]+/[0-9]+) of.*: not freeing it at.*")
free_re = re.compile(r"\[D\^PaRSEC *([0-9]+)\]:.*entry (0x[0-9a-f]+/[0-9]+) of.*: freeing it at.*")

allocations = {}
history = {}

for fn in sys.argv:
    try:
        with open(fn) as fp:
            nballoc = 0
            for line in fp:
                mline = alloc_re.match(line)
                if mline:
                    nballoc+=1
                    entry_key = mline.group(1) + "_" + mline.group(2)
                    if entry_key in allocations:
                        print("Allocation (A)\n  " + mline.group())
                        print("is returning the same value as Allocation (B)\n  " + allocations[entry_key])
                        print("that was not freed yet?!?")
                        print("History of Allocation (B) follows:\n" + history[entry_key])
                        sys.exit(1)
                    allocations[entry_key] = mline.group()
                    history[entry_key] = "  " + mline.group()
                else:
                    mline = notfree_re.match(line)
                    if mline:
                        entry_key = mline.group(1) + "_" + mline.group(2)
                        if entry_key in allocations:
                            history[entry_key] += "  " + mline.group()
                        else:
                            print("Reference on Allocation\n  " + mline.group())
                            if entry_key in history:
                                print("has no initial allocation line: doing something on an entry that was already freed with following history:\n"+
                                      history[entry_key])
                            else:
                                print("has no initial allocation line: doing something on an entry that was never allocated\n")
                            sys.exit(1)
                    else:
                        mline = free_re.match(line)
                        if mline:
                            entry_key = mline.group(1) + "_" + mline.group(2)
                            if entry_key in allocations:
                                if entry_key in history:
                                    history[entry_key] += mline.group()
                                    allocations.pop(entry_key, None)
                                else:
                                    print("This should never happen: Allocation\n  " + mline.group())
                                    print("is freed, and it has been added to the allocations dictionary, but not in the history dictionary.")
                                    sys.exit(2)
                            else:
                                if entry_key in history:
                                    print("Allocation\n" + mline.group())
                                    print("Has been already freed. We tried to free it again. History is following:\n" + history[entry_line])
                                    sys.exit(1)
                                else:
                                    print("Allocation\n  " + mline.group())
                                    print("is freed while it was never allocated before.")
                                    sys.exit(1)
            print(str(nballoc) + " allocations found in trace " + fn)
            if len(allocations) > 0:
                print("There are " + str(len(allocations)) + " allocations that have not been freed. Their history follows:\n")
                i=0
                for key, value in allocations.iteritems():
                    print(str(i) + ":")
                    i+=1
                    if key in history:
                        print(history[key])
                    else:
                        print("  Has no history (this should never happen)")
                    print("\n")
            else:
                print("No issue found with trace " + fn )

    except IOError:
        print("Could not open " + fn)
