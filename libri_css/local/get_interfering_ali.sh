#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
# End configuration section

stage=0
nj=40

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
data_dir=$2
lang=$3
src_dir=$4
ali_dir=$5

new_data_dir=${data_dir}_interfering

set -e -o pipefail

if ! [ -f $info_file ]; then
  echo "$0: $info_file does not exist. Cannot proceed further."
  exit 1
fi

if [ $stage -le 0 ]; then
  # Create a data directory containing only interfering utterance segments
  mkdir -p $new_data_dir
  cp $data_dir/wav.scp $new_data_dir/

  cat $info_file | python -c "import sys

lines = sys.stdin.readlines()
for line in lines:
  _, uttid, _, _, start, end, text = line.strip().split(' ', 6)
  start = float(start)
  end = float(end)
  segid = '{0}-{1:06d}-{2:06d}'.format(uttid, int(start*100), int(end*100))
  print ('{} {} {} {}'.format(segid, uttid, start, end))
" > $new_data_dir/segments

  paste -d" " <( cut -d" " -f1 $new_data_dir/segments ) <( cut -d" " -f7- $info_file ) \
    > $new_data_dir/text

  paste -d" " <( cut -d" " -f1 $new_data_dir/segments ) <( cut -d"-" -f1 $new_data_dir/segments ) \
    > $new_data_dir/utt2spk

  utils/utt2spk_to_spk2utt.pl $new_data_dir/utt2spk > $new_data_dir/spk2utt
  utils/fix_data_dir.sh $new_data_dir
fi

if [ $stage -le 1 ]; then
  # Extract features for the interfering segments
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf \
    $new_data_dir
  steps/compute_cmvn_stats.sh $new_data_dir
  utils/fix_data_dir.sh $new_data_dir
fi

if [ $stage -le 2 ]; then
  # Generate partial alignments for the interfering segments
  steps/align_fmllr.sh --nj $nj --cmd "$decode_cmd"  \
    $new_data_dir $lang $src_dir $ali_dir
fi

exit 0