#!/bin/bash

# This is a recipe for target speaker ASR on the LibriCSS data. For creating
# the training alignments, we will use the tri5b model trained on 
# Librispeech data. Corresponding training stages have not been provided here.

librispeech_corpus=/export/corpora5/LibriSpeech #/PATH/TO/LIBRISPEECH/data
libricss_corpus=/export/fs01/LibriCSS
simlibricss_corpus=/export/fs01/sim-LibriCSS

data=./corpus
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

mfccdir=mfcc
fbankdir=fbank

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
chaindir=exp/chain_sub${subsampling}
model_dirname=model_blstm_ivec
checkpoint=180_220.mdl
resume=

data_affix=_tsmix
train_set=train_sim${data_affix}
test_sets="dev_sim${data_affix} test_sim${data_affix}"

nj=40
decode_nj=40
num_split=20

score_opts="--min-lmwt 6 --max-lmwt 13"
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
mkdir -p $data

# First prepare the simulated LibriCSS data in the Kaldi data format.
if [ $stage -le 0 ]; then
  local/data_prep_sim.sh --data-affix "${data_affix}" $simlibricss_corpus $librispeech_corpus
fi

# We will use the clean utterances for both alignments as well as i-vector
# extraction. So we need both 13-dim (for alignments) and 40-dim MFCCs
# (for i-vectors).
if [ $stage -le 1 ]; then
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/fs{01,02,03}/$USER/kaldi-data/egs/libri_css/s5/$mfcc/storage \
     $mfccdir/storage
  fi

  # Prepare new data directory which will be used for i-vector extraction
  for dataset in ${train_set} ${test_sets}; do
    mkdir -p data/${dataset}_clean
    cp data/${dataset}/{utt2spk,wav_clean.scp,spk2utt,text} data/${dataset}_clean
    mv data/${dataset}_clean/wav_clean.scp data/${dataset}_clean/wav.scp
    cp -r data/${dataset}_clean data/${dataset}_clean_hires

    # Obtain features for alignments
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf \
      data/${dataset}_clean exp/make_mfcc/${dataset}_clean $mfccdir
    steps/compute_cmvn_stats.sh data/${dataset}_clean
    utils/fix_data_dir.sh data/${dataset}_clean

    # Obtain features for i-vector extraction
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf \
      data/${dataset}_clean_hires exp/make_mfcc/${dataset}_clean_hires $mfccdir
    steps/compute_cmvn_stats.sh data/${dataset}_clean_hires
    utils/fix_data_dir.sh data/${dataset}_clean_hires
  done
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_clean data/lang exp/tri5b exp/tri5b_ali_${train_set}_clean
fi

if ! [ -d exp/tri5b/graph_tgsmall ]; then
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri5b exp/tri5b/graph_tgsmall
fi

if [ $stage -le 3 ]; then
  # Extract i-vectors for training data. Use training alignments for silence weighting
  local/nnet3/target_speaker/extract_ivectors_oracle.sh --nj $decode_nj --mfccdir $mfccdir \
    data/${train_set}_clean_hires exp/tri5b_ali_${train_set}_clean
  
  # Extract i-vectors for test sets. We need to first decode to get alignments 
  # for silence weighting.
  for dataset in ${test_sets}; do
    echo "$0: decoding enrolment utterances for silence weighting"
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --skip-scoring true \
      exp/tri5b/graph_tgsmall data/${dataset}_clean exp/tri5b/decode_tgsmall_${dataset}_clean || exit 1
    local/nnet3/target_speaker/extract_ivectors_oracle.sh --nj $decode_nj --mfccdir $mfccdir \
      data/${dataset}_clean_hires exp/tri5b/decode_tgsmall_${dataset}_clean
  done
fi

if [ $stage -le 4 ]; then
  traindir=data/${train_set}
  feat_affix=_fbank
  utils/copy_data_dir.sh ${traindir} ${traindir}${feat_affix}
  steps/make_fbank.sh --cmd "$train_cmd" --nj $nj ${traindir}${feat_affix} \
    exp/make_fbank/${train_set} $fbankdir
  utils/fix_data_dir.sh ${traindir}${feat_affix}
  steps/compute_cmvn_stats.sh ${traindir}${feat_affix}
  utils/fix_data_dir.sh ${traindir}${feat_affix}
fi

if [ $stage -le 5 ]; then
  lang=data/lang_chain

  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${subsampling} \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 7000 data/${train_set}_clean \
    $lang exp/tri5b_ali_${train_set}_clean ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst
fi

if [ $stage -le 6 ]; then
  split_memmap_data.sh data/${train_set}_fbank ${num_split} 
  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:data/${train_set}_fbank/pdfid.${subsampling}.tgt
fi

if [ $stage -eq 7 ]; then
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)

  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi 
  ivector_dir=exp/nnet3_cleaned/ivectors_${train_set}_clean_hires
  ivector_dim=$(feat-to-dim scp:${ivector_dir}/ivectors.scp - || exit 1;)
  idim=$(feat-to-dim scp:data/${train_set}_fbank/feats.scp - || exit 1;)

  train_async_parallel.sh ${resume_opts} \
    --gpu true \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainBLSTMWithIvector \
    --datasetname HybridASRWithIvector \
    --hdim 1024 \
    --num-layers 6 \
    --idim ${idim} \
    --ivector-dim ${ivector_dim} \
    --dropout 0.2 \
    --prefinal-dim 512 \
    --warmup 20000 \
    --decay 1e-07 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.0002 \
    --batches-per-epoch 500 \
    --num-epochs 300 \
    --nj 4 \
    "[ \
        {\
    'data': 'data/${train_set}_fbank', \
    'tgt': 'data/${train_set}_fbank/pdfid.${subsampling}.tgt', \
    'ivectors': 'exp/nnet3_cleaned/ivectors_${train_set}_clean_hires/ivectors.scp', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5,
    'mean_norm':True, 'var_norm':'norm'
        }\
     ]" \
    `dirname ${chaindir}`/${model_dirname}
fi
exit 1
