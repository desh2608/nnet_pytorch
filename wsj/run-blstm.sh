#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
traindir=data/train_si284
feat_affix=_fbank64
chaindir=exp/chain_sub3
num_leaves=3500
model_dirname=model_blstm_1a
batches_per_epoch=500
num_epochs=50
train_nj=4
resume=
num_split=10 # number of splits for memory-mapped data for training
average=true

. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
targets=${traindir}${feat_affix}/pdfid.${subsampling}.tgt
trainname=`basename ${traindir}`

if [ $stage -le 1 ]; then
  # Extract 64-dim filterbank features for nnet training
  # utils/copy_data_dir.sh ${traindir} ${traindir}${feat_affix}
  steps/make_fbank.sh --cmd "$train_cmd" --nj 20 ${traindir}${feat_affix} || exit 1;
  steps/compute_cmvn_stats.sh ${traindir}${feat_affix} || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "Creating Chain Topology, Denominator Graph, and nnet Targets ..."
  lang=data/lang_chain
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo

  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${subsampling} \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" ${num_leaves} ${traindir} \
    $lang exp/tri4b_ali_si284 ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst

  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:${targets}
fi

if [ $stage -le 3 ]; then
  echo "Dumping memory mapped features ..."
  split_memmap_data.sh ${traindir}${feat_affix} ${targets} ${num_split} 
fi

# Multigpu training of Chain-BLSTM with optimizer state averaging
if [ $stage -eq 4 ]; then
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)

  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi 
  idim=$(feat-to-dim scp:${traindir}${feat_affix}/feats.scp - || exit 1;)

  train_async_parallel.sh ${resume_opts} \
    --gpu true \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainBLSTM \
    --datasetname HybridASR \
    --hdim 512 \
    --num-layers 3 \
    --idim ${idim} \
    --dropout 0.2 \
    --prefinal-dim 512 \
    --warmup 6000 \
    --decay 1e-07 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.0002 \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs ${num_epochs} \
    --nj ${train_nj} \
    "[ \
        {\
    'data': '${traindir}${feat_affix}', \
    'tgt': '${traindir}${feat_affix}/pdfid.${subsampling}.tgt', \
    'batchsize': 128, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5,
    'mean_norm':True, 'var_norm':'norm'
        }\
     ]" \
    `dirname ${chaindir}`/${model_dirname}
fi

# Average the last 60 epochs
if $average; then
  echo "Averaging the last few epochs ..."
  average_models.py `dirname ${chaindir}`/${model_dirname} 64 40 50
fi