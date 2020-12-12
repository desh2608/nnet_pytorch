#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
chaindir=exp/chain_sub3
model_dirname=model_blstm_1a
checkpoint=50.mdl
acwt=1.0
test_sets="test_dev93 test_eval92"
decode_nj=80

. ./utils/parse_options.sh

set -euo pipefail

tree_dir=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`

## Make graph if it does not exist
if [ $stage -le 0 ]; then
  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr/phones.txt data/lang_chain/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgpr \
    $tree_dir $tree_dir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_test_bd_tgpr/phones.txt data/lang_chain/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_bd_tgpr \
    $tree_dir $tree_dir/graph_bd_tgpr || exit 1;
fi

## Prepare the test sets if not already done
if [ $stage -le 1 ]; then
  local/prepare_test.sh --subsampling ${subsampling} --test-sets $test_sets \
    --nj $decode_nj
fi

if [ $stage -le 2 ]; then
  # Averaging models from epochs 40 to 50
  average_models.py `dirname ${chaindir}`/${model_dirname} 64 40 50
fi

if [ $stage -le 3 ]; then
  for ds in $test_sets; do 
    # Decoding with pruned 3-gram language models
    for lm in tgpr bd_tgpr; do
      decode_nnet_pytorch.sh --checkpoint ${checkpoint} \
                            --acoustic-scale ${acwt} \
                            --post-decode-acwt ${post_decode_acwt} \
                            --nj ${decode_nj} \
                            --score false \
                            data/${ds}_fbank exp/${model_dirname} \
                            ${tree_dir}/graph_${lm} exp/${model_dirname}/decode_${checkpoint}_graph_${lm}_${acwt}_${ds}
      
      echo ${decode_nj} > exp/${model_dirname}/decode_${checkpoint}_graph_${lm}_${acwt}_${ds}/num_jobs

      steps/score_kaldi.sh --cmd "$decode_cmd" \
        --min-lmwt 6 --max-lmwt 18 --word-ins-penalty 0.0 \
        data/${ds}_fbank ${tree_dir}/graph_${lm} exp/${model_dirname}/decode_${checkpoint}_graph_${lm}_${acwt}_${ds}
    done
    
    # Rescoring with 4-gram LM
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test_bd_{tgpr,fgconst} \
      data/${ds}_fbank exp/${model_dirname}/decode_${checkpoint}_graph_bd_tgpr_${acwt}_${ds} \
      exp/${model_dirname}/decode_${checkpoint}_graph_fgconst_${acwt}_${ds} || exit 1
  done
fi
