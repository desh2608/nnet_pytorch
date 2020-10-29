#! /usr/bin/env python3
# Copyright   2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

"""This script prints information about the auxiliary (non-target)
speech for each target utterance in the following format:
<target-uttid> <sil-dur> <other-segid1> <sil-dur>
"""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script takes a data directory where the recordings may
                    contain overlapped speech. For each utterance, it then obtains
                    transcript corresponing to the interfering speech regions. It
                    takes as input a data directory containing a segments file, which
                    is used to obtain the time marks for the interfering regions,
                    and a CTM file which is used to get the text corresponding to the
                    interfering region.""")

    parser.add_argument("utt2dur", type=str,
                        help="File containing uttids and durations")
    parser.add_argument("interfering_segments", type=str,
                        help="Segments file containing interfering segments")
    parser.add_argument("aux_info", type=str,
                        help="Information file created from get_interfering_speech.py")
    parser.add_argument("--frame-shift", dest="frame_shift", type=float, default=0.01,
                        help="Frame shift duration for MFCC extraction")

    args = parser.parse_args()

    return args


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

class InterferingPair:
    def __init__(self, parts):
        self.target_utt = parts[0]
        self.interfering_utt = parts[1]
        self.target_start = float(parts[2])
        self.target_end = float(parts[3])
        self.interfering_start = float(parts[4])
        self.interfering_end = float(parts[5])
        self.interfering_seg_id = "{}-{:06d}-{:06d}".format(self.interfering_utt,
            int(100*self.interfering_start), int(100*self.interfering_end))

def get_aux_info(target_durations, utt_to_interfering_segs):
    target_to_aux_info = {}
    for target_utt in target_durations:
        total_dur = target_durations[target_utt]
        if target_utt not in utt_to_interfering_segs or len(utt_to_interfering_segs[target_utt]) == 0:
            target_to_aux_info[target_utt] = [str(total_dur)]
        else:
            interfering_segs = sorted(utt_to_interfering_segs[target_utt], key=lambda x:x.target_start)
            cur_start = 0
            target_to_aux_info[target_utt] = []
            while (len(interfering_segs) > 0):
                next_seg = interfering_segs[0]
                if cur_start < next_seg.target_start:
                    target_to_aux_info[target_utt].append(str(next_seg.target_start - cur_start))
                target_to_aux_info[target_utt].append(next_seg.interfering_seg_id)
                cur_start = next_seg.target_end
                interfering_segs = interfering_segs[1:]
            if (cur_start < total_dur):
                target_to_aux_info[target_utt].append(str(total_dur - cur_start))
    return target_to_aux_info

def main():
    args = get_args()

    # Get list of interfering segment ids
    interfering_segment_ids = []
    with open(args.interfering_segments, 'r') as f:
        for line in f:
            seg_id,_ = line.strip().split(maxsplit=1)
            interfering_segment_ids.append(seg_id)

    # Get list of all pairs of target and interfering segments
    interfering_pairs = []
    with open(args.aux_info, 'r') as f:
        for line in f:
            parts = line.strip().split()
            interfering_pairs.append(InterferingPair(parts))

    # Get durations of all target utterances
    target_durations = {}
    with open(args.utt2dur, 'r') as f:
        for line in f:
            uttid, dur = line.strip().split()
            target_durations[uttid] = float(dur)

    utt_to_interfering_segs = defaultdict(list,
        {utt_id : list(g) for utt_id, g in groupby(interfering_pairs, lambda x: x.target_utt)})

    # For each target utterance, get auxiliary speech info
    target_to_aux_info = get_aux_info(target_durations, utt_to_interfering_segs) 

    for target_utt in target_to_aux_info:
        aux_info = " ".join(target_to_aux_info[target_utt])
        print ("{} {}".format(target_utt, aux_info))
    


if __name__ == '__main__':
    main()
