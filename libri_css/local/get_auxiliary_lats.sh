#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
# End configuration section

stage=2
nj=40
frame_shift=0.01
cmd=run.pl

. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 5 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <info-file> <data-dir> <lang-dir> <src-dir> <lat-dir>"
  echo -e >&2 "eg:\n  $0 data/info data/train data/lang exp/lats"
  exit 1
fi

info_file=$1
utt2dur=$2
lang=$3
interfering_lats_dir=$4
lats_dir=$5

mkdir -p $lats_dir

set -e -o pipefail

if [ $stage -le 0 ]; then
  # First combine all lattices and unzip
  cat $interfering_lats_dir/lat.*.gz | gunzip -c - > $interfering_lats_dir/lats
fi

if [ $stage -le 1 ]; then
  # Read utterances from utt2dur, get interfering segments from
  # aux_info file, and create an info file for auxiliary lattice
  # generation. This file will contain the following:
  # target-uttid <sil1> <interfering-segid> <sil2>
  # where <sil1> and <sil2> are padding according to the 
  # target utterance duration.
  local/get_auxiliary_speech.py --frame-shift $frame_shift $utt2dur \
    $interfering_lats_dir/segments $info_file > $lats_dir/aux_lats_info
fi

if [ $stage -le 2 ]; then
  # Get id of SPN phone
  phone_id=$( awk '/SPN/{ print NR-1; exit }' $lang/phones/silence.txt )
  lattice-generate-aux ark:$interfering_lats_dir/lats $lats_dir/aux_lats_info \
    ark:$lats_dir/lats
fi

exit 0