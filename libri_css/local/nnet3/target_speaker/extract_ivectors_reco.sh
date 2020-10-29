#!/usr/bin/env bash
# Copyright      2020   Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.

set -e -o pipefail

# This script is called from run_ts.sh. It extracts i-vectors for the data
# from the oracle (enrollment) utterances. For each speaker, an enrollment
# utterance is picked randomly such that it is not present in the given
# recording.
nj=50
decode_nj=40
stage=2
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
ivector_extractor=exp/nnet3${nnet3_affix}/extractor
ivector_type=reco
rttm_file= # if provided, it will be used to downweight overlapping segments
mfccdir=mfcc

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <src-dir>"
  echo "e.g.: $0 data/train exp/chain/tdnn_1a"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_dir=$1
src_dir=$2

ivector_affix="_$ivector_type"
name=`basename $data_dir`

graph_dir=$src_dir/graph_tgsmall
if ! [ -d $graph_dir ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test_tgsmall $src_dir $graph_dir
fi

if [ $stage -le 1 ]; then
  echo "$0: decoding oracle utterances for silence weighting"
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $decode_nj --cmd "$decode_cmd" --skip-scoring true \
      $graph_dir $data_dir $src_dir/decode_${name}${ivector_affix}_tgsmall || exit 1
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting iVectors for training data"
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${name}${ivector_affix}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/fs{01,02,03,04}/$USER/kaldi-data/ivectors/libri_css-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi

  if [ -z "$rttm_file" ]; then
    # Note that we only need the lang dir to get the list of silence phones, to
    # downweight them for i-vector estimation. We choose ivector-type "fixed",
    # which means that we extract utterance-wise ivectors for each utterance.
    local/nnet3/target_speaker/internal/extract_ivectors.sh --cmd "$train_cmd" \
      --nj $decode_nj --ivector-type all \
      $data_dir data/lang_test_tgsmall $ivector_extractor \
      $src_dir/decode_${name}${ivector_affix}_tgsmall $ivectordir || exit 1;
  else
    # We first get the overlap weights file. We choose ivector-type "all",
    # which means that we extract speaker-level ivectors. If you are extracting
    # i-vectors for training data, it may be better to use ivector-type "fixed".
    echo "$0: Using provided RTTM file to compute weights file"
    ivector_overlap_weights=${ivectordir}/overlap_weights.gz
    local/overlap/get_overlap_segments.py $rttm_file | grep "overlap" |\
      local/extract_overlap_weights.py --overlap-weight 0.0 ${data_dir}/segments \
      ${data_dir}/utt2num_frames - | sort -u | gzip -c > $ivector_overlap_weights 
    
    local/nnet3/target_speaker/internal/extract_ivectors.sh --cmd "$train_cmd" \
      --nj $decode_nj --ivector-type fixed --overlap-weights-file $ivector_overlap_weights \
      $data_dir data/lang_test_tgsmall $ivector_extractor \
      $src_dir/decode_${name}${ivector_affix}_tgsmall $ivectordir || exit 1;
  fi
fi

exit 0;
