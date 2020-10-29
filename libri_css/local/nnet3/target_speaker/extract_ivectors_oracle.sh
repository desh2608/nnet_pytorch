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
stage=0
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
ivector_extractor=exp/nnet3${nnet3_affix}/extractor
ivector_affix=
mfccdir=mfcc

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <ali-dir>"
  echo "e.g.: $0 data/train exp/tri3/decode_dev"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_dir=$1
ali_dir=$2

name=`basename $data_dir`

for f in $data_dir/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

echo "$0: extracting iVectors for training data"
ivectordir=exp/nnet3${nnet3_affix}/ivectors_${name}${ivector_affix}
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
  utils/create_split_dir.pl /export/fs{01,02,03,04}/$USER/kaldi-data/ivectors/libri_css-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
fi

# Note that we only need the lang dir to get the list of silence phones, to
# downweight them for i-vector estimation. We choose ivector-type "shuffle",
# which means that we extract utterance-wise ivectors from a different 
# utterance of the same speaker.
local/nnet3/target_speaker/internal/extract_ivectors.sh --cmd "$decode_cmd" \
  --nj $decode_nj --ivector-type shuffle \
  $data_dir data/lang_test_tgsmall $ivector_extractor \
  $ali_dir $ivectordir || exit 1;

exit 0;
