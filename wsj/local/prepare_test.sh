#!/bin/bash

test_sets="test_dev93 test_eval92"
subsampling=3
nj=40

. ./cmd.sh
. ./path.sh

. ./utils/parse_options.sh

set -euo pipefail

for dataset in $test_sets; do
  echo "-------------- Making ${dataset} ----------------------"
  utils/copy_data_dir.sh data/${dataset} data/${dataset}_fbank
  steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
    data/${dataset}_fbank
  utils/fix_data_dir.sh data/${dataset}_fbank
  steps/compute_cmvn_stats.sh data/${dataset}_fbank
  utils/fix_data_dir.sh data/${dataset}_fbank

  python ../nnet_pytorch/utils/prepare_unlabeled_tgt.py --subsample ${subsampling} \
    data/${dataset}_fbank/utt2num_frames > data/${dataset}_fbank/pdfid.${subsampling}.tgt
  ../nnet_pytorch/utils/split_memmap_data.sh data/${dataset}_fbank data/${dataset}_fbank/pdfid.${subsampling}.tgt 5
done

exit 0;
 

