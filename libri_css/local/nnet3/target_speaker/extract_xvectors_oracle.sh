#!/usr/bin/env bash
# Copyright      2020   Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.

set -e -o pipefail

# This script is called from run_ts.sh. It extracts x-vectors for the data
# from the oracle (enrollment) utterances. For each speaker, an enrollment
# utterance is picked randomly such that it is not present in the given
# recording.
nj=40
decode_nj=40
stage=0
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
xvector_affix=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <xvector-extracor>"
  echo "e.g.: $0 data/train exp/xvector_nnet_1a"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_dir=$1
xvector_extractor=$2

name=`basename $data_dir`

for f in $data_dir/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  echo "$0: extracting x-vectors for data"
  xvectordir=exp/nnet3${nnet3_affix}/xvectors_${name}${xvector_affix}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $xvectordir/storage ]; then
    utils/create_split_dir.pl /export/fs{01,02,03,04}/$USER/kaldi-data/xvectors/libri_css-$(date +'%m_%d_%H_%M')/s5/$xvectordir/storage $xvectordir/storage
  fi

  # We choose xvector-type "shuffle", which means that we extract utterance-wise 
  # xvectors from a different utterance of the same speaker.
  local/nnet3/target_speaker/internal/extract_xvectors.sh --cmd "$decode_cmd" \
    --nj $nj --xvector-type shuffle \
    $data_dir $xvector_extractor $xvectordir || exit 1;
fi

exit 0;
