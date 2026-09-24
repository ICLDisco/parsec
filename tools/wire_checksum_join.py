#!/usr/bin/env python3
#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#
"""Join the WIRE SEND and WIRE RECV records a PARSEC_DEBUG_WIRE_CHECKSUM run emits.

Each transfer is fingerprinted twice, once by the sender just before the
bytes are handed to the transport and once by the receiver after they have
landed. The two records share a key -- sending rank, receiving rank, the
sender's deps pointer and the flow index -- so joining them says whether the
transport delivered what was given to it.

    mpiexec ... 2>&1 | tee run.log
    wire_checksum_join.py run.log

The seed is a fixed constant and the generator is a function of the tile
coordinates, so every run moves the same bytes for the same tile. Two runs
can therefore be compared directly, which locates the first tile whose
contents differ rather than merely confirming a transfer was faithful.

Transfers are compared by tile, never by task: a DTD task is named by a
counter handed out as tasks are inserted, so under untied insertion the same
task wears a different number in every run and comparing by task reports the
whole run as different.

Only sends are compared. A received copy is a fresh buffer with no data
collection behind it yet, so it cannot name its tile; the within-run join
already establishes that each receive matched its send, which makes the
sends a faithful stand-in for both.

A send whose copy has no data collection either is reported as anonymous
and left out of the comparison, since every such record would fall in the
same bucket and compare unequal for no reason. Their number is printed so
the blind spot stays visible.

    wire_checksum_join.py --compare good.log bad.log

A transfer log only shows data that crossed a rank boundary. A tile that is
updated in place by local tasks and ends up wrong never appears there. With
--mca pins_data_checksum_trace 1 the runtime also fingerprints both ends of
every task, and --compare-tiles finds the first tile whose set of observed
values differs, local tasks included:

    wire_checksum_join.py --compare-tiles good.log bad.log

Reports transfers whose two fingerprints disagree, and receives that have no
matching send. A send with no matching receive is normal at the tail of a
run, where output may be cut off, so those are only counted.

A send is paired with its receive by the task that emitted it, qualified by
its taskpool. A DTD task is numbered by an atomic counter as it is inserted,
so that pair is unique for the length of a run. The sender's deps pointer
would be the obvious choice and is wrong: those structures come from a free
list, and one address names many different transfers over a run.
"""

import re
import sys
from collections import OrderedDict, defaultdict, deque

# A tile renders as "A(3, 5)", spaces included, so the name runs up to the k=
# field rather than to the next space.
RECORD = re.compile(
    r'WIRE (SEND|RECV) src=(\d+) dst=(\d+) tile=(.+?) k=(\d+) '
    r'len=(\d+) hash=([0-9a-f]+) tp=(\d+) task=(.+)$')


def read_transfers(path):
    """Every record keyed the way another run would key it: the deps pointer
    is an address and varies, the tile and flow do not. The position of the
    first record under a key is kept so differences can be reported in the
    order the reference run produced them, which is far more useful than
    alphabetical when hunting for the one that came first."""
    seen = defaultdict(list)
    first = {}
    anonymous = [0]
    with open(path, errors='replace') as handle:
        for position, line in enumerate(handle):
            m = RECORD.search(line)
            if None is m:
                continue
            kind, src, dst, tile, k, length, digest, _tp, task = m.groups()
            if 'SEND' != kind:
                continue
            if 'anonymous' == tile:
                anonymous[0] += 1
                continue
            key = (kind, int(src), int(dst), tile, int(k))
            seen[key].append((int(length), digest, task))
            first.setdefault(key, position)
    return seen, first, anonymous[0]


def compare(good_path, bad_path):
    good, order, good_anon = read_transfers(good_path)
    bad, bad_order, bad_anon = read_transfers(bad_path)
    if not good or not bad:
        print('no WIRE records in one of the logs')
        return 1

    def when(key):
        return order.get(key, bad_order.get(key, 1 << 60))

    differing, missing = [], []
    for key in sorted(set(good) | set(bad), key=when):
        want, got = good.get(key, []), bad.get(key, [])
        # The task is shown for context but never compared: its number comes
        # from insertion order and differs between runs.
        if [r[:2] for r in want] == [r[:2] for r in got]:
            continue
        if not want or not got:
            missing.append((key, len(want), len(got)))
        else:
            differing.append((key, want, got))

    print('%d sends in %s, %d in %s (%d and %d anonymous, not compared)'
          % (sum(len(v) for v in good.values()), good_path,
             sum(len(v) for v in bad.values()), bad_path, good_anon, bad_anon))
    print('%d differ, %d appear in only one run; '
          'listed in the order the reference run produced them\n'
          % (len(differing), len(missing)))

    for (kind, src, dst, tile, k), want, got in differing[:40]:
        print('DIFFERS %s tile %s flow %d src=%d dst=%d' % (kind, tile, k, src, dst))
        for label, records in (('good', want), ('bad ', got)):
            print('    %s %s' % (label,
                  ' '.join('%s[%s]' % (d, t) for _l, d, t in records)))
    if len(differing) > 40:
        print('... and %d more' % (len(differing) - 40))
    for (kind, src, dst, tile, k), nwant, ngot in missing[:20]:
        print('ONLY IN ONE RUN %s tile %s flow %d src=%d dst=%d: %d good, %d bad'
              % (kind, tile, k, src, dst, nwant, ngot))
    if len(missing) > 20:
        print('... and %d more present in only one run' % (len(missing) - 20))

    if differing or missing:
        print('\nthe first entry above is the earliest transfer known to carry '
              'different bytes;\nwhatever produced that tile read something that '
              'already differed, so look at its inputs')
        return 1
    print('the two runs moved identical bytes for every tile')
    return 0


def main(paths):
    sends, recvs = {}, OrderedDict()
    duplicates = []
    nb_sends = nb_recvs = 0
    for path in paths:
        with open(path, errors='replace') as handle:
            for line in handle:
                m = RECORD.search(line)
                if None is m:
                    continue
                kind, src, dst, _tile, k, length, digest, tp, task = m.groups()
                key = (int(src), int(dst), int(tp), task.strip(), int(k))
                if 'SEND' == kind:
                    if key in sends:
                        duplicates.append(key)
                    sends[key] = (int(length), digest)
                    nb_sends += 1
                else:
                    recvs[key] = (int(length), digest)
                    nb_recvs += 1

    if not nb_sends and not nb_recvs:
        print('no WIRE records found; is the build PARSEC_DEBUG_WIRE_CHECKSUM=ON?')
        return 1

    corrupted, orphan, matched = [], [], 0
    for key, (length, digest) in recvs.items():
        if key not in sends:
            orphan.append(key)
            continue
        sent_length, sent_digest = sends[key]
        matched += 1
        if sent_digest != digest or sent_length != length:
            corrupted.append((key, (sent_length, sent_digest), (length, digest)))

    print('%d sends, %d receives, %d matched' % (nb_sends, nb_recvs, matched))

    if duplicates:
        print('%d keys occur more than once; the task should be unique within '
              'a run, so the pairing below cannot be trusted' % len(duplicates))
    for key, sent, got in corrupted:
        src, dst, tp, task, k = key
        print('CORRUPTED src=%d dst=%d tp=%d %s flow %d: sent %d bytes %s, '
              'received %d bytes %s' % (src, dst, tp, task, k,
                                        sent[0], sent[1], got[0], got[1]))
    for src, dst, tp, task, k in orphan[:20]:
        print('UNMATCHED receive src=%d dst=%d tp=%d %s flow %d' % (src, dst, tp, task, k))
    if len(orphan) > 20:
        print('... and %d more unmatched receives' % (len(orphan) - 20))

    if corrupted:
        print('\n%d transfer(s) delivered something other than what was sent'
              % len(corrupted))
        return 1
    print('every matched transfer delivered exactly what was sent')
    return 0




TILE_RECORD = re.compile(
    r'TILE (IN |OUT) tile=(.+?) flow=(\d+) hash=([0-9a-f]+) tp=(\d+) task=(.+)$')


def read_tiles(path):
    """Group every value a tile was seen holding, by tile and direction.

    Values are collected as a set rather than a sequence: tasks run in
    whatever order the scheduler picks, so the same tile is read in a
    different order from run to run even when nothing is wrong. What must
    match is which values existed, not when each was observed.
    """
    seen, order, tasks = defaultdict(set), {}, defaultdict(set)
    with open(path, errors='replace') as handle:
        for position, line in enumerate(handle):
            m = TILE_RECORD.search(line)
            if None is m:
                continue
            direction, tile, _flow, digest, _tp, task = m.groups()
            key = (direction.strip(), tile)
            seen[key].add(digest)
            tasks[(key, digest)].add(task.strip())
            order.setdefault(key, position)
    return seen, order, tasks


def compare_tiles(good_path, bad_path):
    good, order, good_tasks = read_tiles(good_path)
    bad, bad_order, bad_tasks = read_tiles(bad_path)
    if not good or not bad:
        print('no TILE records; run with --mca pins_data_checksum_trace 1')
        return 1

    differing = [k for k in sorted(set(good) | set(bad),
                                   key=lambda k: order.get(k, bad_order.get(k, 1 << 60)))
                 if good.get(k, set()) != bad.get(k, set())]

    print('%d tiles observed in %s, %d in %s; %d differ, in the order the '
          'reference run first saw them\n'
          % (len(good), good_path, len(bad), bad_path, len(differing)))

    for key in differing[:30]:
        direction, tile = key
        only_good = good.get(key, set()) - bad.get(key, set())
        only_bad = bad.get(key, set()) - good.get(key, set())
        print('DIFFERS %s tile %s' % (direction, tile))
        for label, digests, where in (('only good', only_good, good_tasks),
                                      ('only bad ', only_bad, bad_tasks)):
            for digest in sorted(digests):
                print('    %s %s  %s' % (label, digest,
                                         ', '.join(sorted(where[(key, digest)]))))
    if len(differing) > 30:
        print('... and %d more' % (len(differing) - 30))

    if differing:
        print('\nthe first entry is the earliest tile to hold a value the good '
              'run never held;\nthe task listed against it is the one that read '
              'or produced it')
        return 1
    print('every tile held the same set of values in both runs')
    return 0


if '__main__' == __name__:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if '--compare-tiles' == sys.argv[1]:
        if 4 != len(sys.argv):
            sys.exit('--compare-tiles takes exactly two logs')
        sys.exit(compare_tiles(sys.argv[2], sys.argv[3]))
    if '--compare' == sys.argv[1]:
        if 4 != len(sys.argv):
            sys.exit('--compare takes exactly two logs')
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1:]))
