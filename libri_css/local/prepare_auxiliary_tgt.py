#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aux_info', type=str)
    parser.add_argument('target_pdfs', type=str)
    parser.add_argument('auxiliary_pdfs', type=str)

    parser.add_argument('--frame-shift', type=float, default=0.01)
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--pad-label', type=int, default=-1)

    args = parser.parse_args()

    padded_aux_tgt = {}
    with open(args.target_pdfs, 'r') as f:
        for l in f:
            utt, pdfs = l.strip().split(maxsplit=1)
            num_frames = len(pdfs.split())
            padded_aux_tgt[utt] = [str(args.pad_label)]*num_frames

    aux_tgts = {}
    with open(args.auxiliary_pdfs, 'r') as f:
        for l in f:
            utt, pdfs = l.strip().split(maxsplit=1)
            aux_tgts[utt] = pdfs.split()[0::args.subsample]

    with open(args.aux_info, 'r') as f:
        for l in f:
            tgt_utt, aux_utt, start_time, end_time, aux_start, aux_end, _ = l.strip().split(maxsplit=6)
            start_idx = int(float(start_time)/(args.frame_shift*args.subsample))
            segid = '{0}-{1:06d}-{2:06d}'.format(aux_utt, int(float(aux_start)*100), int(float(aux_end)*100))
            try:
                aux_tgt = aux_tgts[segid]
                padded_tgts = padded_aux_tgt[tgt_utt]
            except:
                continue 
            
            orig_len = len(padded_tgts)
            end_idx = start_idx + len(aux_tgt)
            padded_tgts[start_idx:end_idx] = aux_tgt
            assert(orig_len == len(padded_tgts))
            padded_aux_tgt[tgt_utt] = padded_tgts

    for utt in padded_aux_tgt.keys():
        tgt = " ".join(padded_aux_tgt[utt])
        print ("{} {}".format(utt,tgt))  

if __name__ == "__main__":
    main()

