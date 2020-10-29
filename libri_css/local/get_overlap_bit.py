#!/usr/bin/env python

# Copyright 2020  Desh Raj (Johns Hopkins University)
# Apache 2.0

# This script performs a similar job as the extract_vad_weights.sh
# except that instead of downweighting silence, it downweights
# overlap frames. The overlap information comes from an "overlap RTTM"
# which may have been obtained by an external overlap detector.

from __future__ import division

import argparse
import numpy as np
import itertools
import sys
from collections import defaultdict

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script performs a similar job as the extract_vad_weights.sh
                    except that instead of downweighting silence, it downweights
                    overlap frames. The overlap information comes from an "overlap RTTM"
                    which may have been obtained by an external overlap detector.
        """)

    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift value in seconds")
    parser.add_argument("--overlap-weight", type=float, default=0.001,
                        help="Weight to be assigned to overlap frames")
    parser.add_argument("--overlap-bit", type=int, default=0,
                        help="Set 1 if using script to get frame-wise overlap bit")
    parser.add_argument("--transpose", action="store_true",
                        help="Paste output as column vector")
    parser.add_argument("segments", type=str,
                        help="""Segments file to be used to output weights
                        """)
    parser.add_argument("utt2num_frames", type=str,
                        help="""The number of frames per reco
                        is used to determine the num-rows of the output matrix
                        """)
    parser.add_argument("overlap_rttm", type=str,
                        help="Input RTTM file containing overlap segments")

    args = parser.parse_args()

    if args.frame_shift < 0.0001 or args.frame_shift > 1:
        raise ValueError("--frame-shift should be in [0.0001, 1]; got {0}"
                         "".format(args.frame_shift))
    return args

class Segment:
    """Stores all information about a segment"""
    reco_id = ''
    spk_id = ''
    start_time = 0
    dur = 0
    end_time = 0

    def __init__(self, reco_id, start_time, dur = None, end_time = None, label = None):
        self.reco_id = reco_id
        self.start_time = start_time
        if (dur is None):
            self.end_time = end_time
            self.dur = end_time - start_time
        else:
            self.dur = dur
            self.end_time = start_time + dur
        self.label = label

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def get_weight_vector(num_frames, start_time, end_time, frame_shift, overlap_segs, ovl_weight, non_ovl_weight):
    weight_vec = non_ovl_weight*np.ones(num_frames)

    # First we get all start and end times in a list
    tokens = []
    for seg in overlap_segs:
        tokens.append(('BEG', seg.start_time))
        tokens.append(('END', seg.end_time))
    tokens += [('BEG', start_time), ('END', end_time)]
    
    # Now we get those segments from current utterance which lie in overlap regions
    cur_overlaps = []
    count = 0
    for token in sorted(tokens, key=lambda x: x[1]):
        if (token[0] == 'BEG'):
            count += 1
            if (count == 2):
                beg_ovl = token[1]
        else:
            count -= 1
            if (count == 1):
                cur_overlaps.append((beg_ovl, token[1]))
    
    # Now for all overlapping segments, set the corresponding frame weight
    for segment in cur_overlaps:
        start_frame = int((segment[0] - start_time)/frame_shift)
        end_frame = min(int((segment[1] - start_time) / frame_shift), num_frames)
        weight_vec[start_frame:end_frame] = ovl_weight

    return weight_vec.tolist()

def run(args):
    # Get all utt to num_frames, which will be used to decide the size of 
    # weight vector for each utt
    utt2num_frames = {}
    with common_lib.smart_open(args.utt2num_frames) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError("Could not parse line {0}".format(line))
            utt2num_frames[parts[0]] = int(parts[1])

    # We read all overlap segments and store as a list of objects
    overlap_segs = []
    with common_lib.smart_open(args.overlap_rttm) as f:
        for line in f.readlines():
            parts = line.strip().split()
            overlap_segs.append(Segment(parts[1], float(parts[3]), dur=float(parts[4]), label=parts[7]))

    # We group the segment list into a dictionary indexed by reco_id
    reco2overlaps = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(overlap_segs, lambda x: x.reco_id)})

    if args.overlap_bit:
        args.overlap_weight = 1
        non_ovl_weight = 0
    else:
        non_ovl_weight = 1
    # Now we read all utts one by one and print the weight vector
    with common_lib.smart_open(args.segments) as f:
        for line in f:
            parts = line.strip().split()
            num_frames = utt2num_frames[parts[0]]
            start_time = float(parts[2])
            end_time = float(parts[3])

            weight_vec = get_weight_vector(num_frames, start_time, end_time, args.frame_shift, 
                reco2overlaps[parts[1]], args.overlap_weight, non_ovl_weight)
            if not args.transpose:
                print ("{} [ {} ]".format(parts[0], ' '.join([str(x) for x in weight_vec])))
            else:
                print ("{} [".format(parts[0]))
                for x in weight_vec:
                    print(str(x))
                print ("]")

def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        raise

if __name__ == "__main__":
    main()