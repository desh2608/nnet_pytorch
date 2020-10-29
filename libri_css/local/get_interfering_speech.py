#! /usr/bin/env python3
# Copyright   2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

"""This script takes a data directory where the recordings may
contain overlapped speech. For each utterance, it then obtains
transcript corresponing to the interfering speech regions. It
takes as input a data directory containing a segments file, which
is used to obtain the time marks for the interfering regions,
and a CTM file which is used to get the text corresponding to the
interfering region. It prints the following to STDOUT:
<target-uttid> <interfering-uttid> <target-start> <target-end> \
    <interfering-start> <interfering-end> <words>
"""

import argparse, os
import itertools
from collections import defaultdict
from collections import namedtuple

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script takes a data directory where the recordings may
                    contain overlapped speech. For each utterance, it then obtains
                    transcript corresponing to the interfering speech regions. It
                    takes as input a data directory containing a segments file, which
                    is used to obtain the time marks for the interfering regions,
                    and a CTM file which is used to get the text corresponding to the
                    interfering region.""")

    parser.add_argument("data_dir", type=str,
                        help="Input data directory containing segments file")
    parser.add_argument("ctm_file", type=str,
                        help="""Input CTM file.
                        The format of the CTM file is
                        <segment-id> <channel-id> <begin-time> """
                        """<duration> <word>""")

    args = parser.parse_args()

    return args


class Word:
    def __init__(self, parts):
        self.utt_id = parts[0]
        self.start_time = float(parts[2]) 
        self.dur = float(parts[3])
        self.end_time = self.start_time + self.dur
        self.text = parts[4]

    def __repr__(self):
        return ("{} {} {}".format(self.start_time, self.end_time, self.text))

class Segment:
    def __init__(self, seg_parts, utt2spk_parts):
        self.seg_id = seg_parts[0]
        self.reco_id = seg_parts[1]
        self.start_time = float(seg_parts[2])
        self.end_time = float(seg_parts[3])
        self.dur = self.end_time - self.start_time
        self.spk_id = utt2spk_parts[1]

class Region:
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def read_segments(data_dir):
    segments_path = os.path.join(data_dir,"segments")
    utt2spk_path = os.path.join(data_dir, "utt2spk")
    segments = []
    with open(segments_path, 'r') as seg_file, open(utt2spk_path, 'r') as utt2spk_file:
        for seg_line, utt2spk_line in zip(seg_file.readlines(), utt2spk_file.readlines()):
            seg_parts = seg_line.strip().split()
            utt2spk_parts = utt2spk_line.strip().split()
            segments.append(Segment(seg_parts, utt2spk_parts))
    return segments

def get_overlap_region(seg1, seg2):
    tokens = []
    for seg in [seg1,seg2]:
        tokens.append(('BEG',seg.start_time))
        tokens.append(('END',seg.end_time))
    ovl_region = Region()
    count = 0
    for token in sorted(tokens, key=lambda x:x[1]):
        if token[0] == 'BEG':
            count += 1
            if (count == 2):
                ovl_region.start = token[1]
        else:
            count -= 1
            if (count == 1):
                ovl_region.end = token[1]
    return ovl_region

def get_overlap_words(words, ovl_region, other_start, target_start):
    # Given an overlap region between a target utterance and an interfering utterance,
    # returns a tuple containing:
    # start time w.r.t. target utterance
    # end time w.r.t. target utterance
    # start time w.r.t. interfering utterance
    # end time w.r.t. interfering utterance
    # overlapping words
    ovl_words = []
    rel_start = ovl_region.start - other_start
    rel_end = ovl_region.end - other_start
    for word in words:
        if word.start_time >= rel_start and word.end_time <= rel_end:
            ovl_words.append(word)
    
    if len(ovl_words) == 0:
        return None
    
    ovl_start = ovl_words[0].start_time + other_start - target_start
    ovl_end = ovl_words[-1].end_time + other_start - target_start
    ovl_text = " ".join([word.text for word in ovl_words])
    return (ovl_start, ovl_end, ovl_words[0].start_time, ovl_words[-1].end_time, ovl_text) 


def get_interfering_words(reco_to_segments, utt_to_words):
    interfering_words = {}
    for reco_id in reco_to_segments:
        segments = reco_to_segments[reco_id]

        # Iterate over all the utterances
        for target_seg in segments:
            seg_id = target_seg.seg_id
            interfering_words[seg_id] = {}
            other_segments = [seg for seg in segments if seg != target_seg]

            # For each utterance, find overlaps with all other utterances in the recording.
            for other_seg in other_segments:

                # Compute the overlapping region. This is in terms of the absolute time
                # marks in the recording
                ovl_region = get_overlap_region(target_seg, other_seg)

                # If there is a non-zero overlap, compute the overlap text from the
                # interfering segment. Note that the CTM for the "other_seg" is in
                # relative time (starting from 0), so we will first add other_seg.start_time
                # to it to get the absolute duration. Then, since we want the time 
                # relative to the target_seg's start time, we will subtract it from
                # the absolute time.
                if ovl_region.start is not None:
                    segment_ovl_words = get_overlap_words(utt_to_words[other_seg.seg_id], 
                        ovl_region, other_seg.start_time, target_seg.start_time)
                    if segment_ovl_words is not None:
                        interfering_words[seg_id][other_seg.seg_id] = segment_ovl_words
    return interfering_words

def main():
    args = get_args()

    # Read segments and utt2spk file
    segments = read_segments(args.data_dir)
    reco_to_segments = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})

    # Read the CTM file and store as a list of Word objects
    ctm_words=[]
    with open(args.ctm_file, 'r') as f:
        for line in f:
            ctm_words.append(Word(line.strip().split()))
    
    # Group the list into a dictionary indexed by reco id
    utt_to_words = defaultdict(list,
        {utt_id : list(g) for utt_id, g in groupby(ctm_words, lambda x: x.utt_id)})

    # Get the list of interfering words for each utterance
    interfering_words = get_interfering_words(reco_to_segments, utt_to_words)
    
    for target_utt in interfering_words:
        for interfering_utt in interfering_words[target_utt]:
            ovl_region = interfering_words[target_utt][interfering_utt]
            print ("{} {} {}".format(target_utt, interfering_utt, " ".join([str(x) for x in list(ovl_region)])))


if __name__ == '__main__':
    main()
