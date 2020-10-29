#!/bin/bash

# This is based almost entirely on the Kaldi Librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
librispeech_corpus=/export/corpora5/LibriSpeech #/PATH/TO/LIBRISPEECH/data
libricss_corpus=/export/fs01/LibriCSS
mfccdir=mfcc
ivectordir=exp/nnet3_cleaned

. ./cmd.sh
. ./path.sh

stage=4
subsampling=3
chaindir=exp/chain_sub${subsampling}
model_dirname=model_blstm_ivec
resume=
checkpoint=160_180.mdl

data_affix=_tsreal
test_sets="dev${data_affix}"
conditions="0L 0S OV10 OV20 OV30 OV40"

acwt=1.0
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`

nj=40
decode_nj=30
num_split=20
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree

# DECODING
if [ $stage -le 1 ]; then
  # Echo Make graph if it does not exist
  if [ ! -f ${tree}/graph_tgsmall/HCLG.fst ]; then 
    ./utils/mkgraph.sh --self-loop-scale 1.0 \
      data/lang_test_tgsmall ${tree} ${tree}/graph_tgsmall
  fi

  if ! [ -d data/eval_tsmix_fbank ]; then
    # Prepare the test sets
    # local/data_prep_mono.sh --data-affix $data_affix \
    #   $libricss_corpus $librispeech_corpus
    for dataset in $test_sets; do
      mv data/${dataset}/segments.bak data/${dataset}/segments
      mv data/${dataset}/utt2spk.bak data/${dataset}/utt2spk
      mv data/${dataset}/text.bak data/${dataset}/text
      utils/utt2spk_to_spk2utt.pl data/$dataset/utt2spk > data/$dataset/spk2utt
    done

    for dataset in $test_sets; do
      echo "-------------- Making ${dataset} ----------------------"
      utils/copy_data_dir.sh data/${dataset} data/${dataset}_fbank
      steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
         data/${dataset}_fbank exp/make_fbank/${dataset} fbank
      utils/fix_data_dir.sh  data/${dataset}_fbank
      steps/compute_cmvn_stats.sh  data/${dataset}_fbank
      utils/fix_data_dir.sh  data/${dataset}_fbank

      python ../nnet_pytorch/utils/prepare_unlabeled_tgt.py --subsample ${subsampling} \
        data/${dataset}_fbank/utt2num_frames > data/${dataset}_fbank/pdfid.${subsampling}.tgt
      ../nnet_pytorch/utils/split_memmap_data.sh data/${dataset}_fbank $num_split
    done
  fi
fi

if [ $stage -le 2 ]; then
  # Prepare new data directory which will be used for i-vector extraction
  for dataset in ${test_sets}; do
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

if [ $stage -le 3 ]; then
  # Extract i-vectors for test sets. We need to first decode to get alignments 
  # for silence weighting.
  for dataset in ${test_sets}; do
    echo "$0: decoding enrolment utterances for silence weighting"
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --skip-scoring true \
      exp/tri5b/graph_tgsmall data/${dataset}_clean exp/tri5b/decode_tgsmall_${dataset}_clean || exit 1
    local/nnet3/target_speaker/extract_ivectors_oracle.sh --decode-nj $decode_nj --mfccdir $mfccdir \
      data/${dataset}_clean_hires exp/tri5b/decode_tgsmall_${dataset}_clean
  done
fi

if [ $stage -le 4 ]; then
  # Average models (This gives better performance)
  # average_models.py `dirname ${chaindir}`/${model_dirname} 64 160 180 
  for ds in $test_sets; do 
    decode_nnet_pytorch.sh --score false \
                           --checkpoint ${checkpoint} \
                           --acoustic-scale ${acwt} \
                           --post-decode-acwt ${post_decode_acwt} \
                           --ivector_scp ${ivectordir}/ivectors_${ds}_clean_hires/ivectors.scp \
                           --nj ${decode_nj} \
                           --save-post true \
                           data/${ds}_fbank exp/${model_dirname} \
                           ${tree}/graph_tgsmall exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}
    
    echo ${decode_nj} > exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}/num_jobs
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" --skip-scoring true \
      data/lang_test_{tgsmall,fglarge} \
      data/${ds}_fbank exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}{,_fglarge_rescored} 

    ./local/score.sh --cmd "$decode_cmd" \
      --min-lmwt 6 --max-lmwt 18 --word-ins-penalty 0.0 \
      data/${ds}_fbank ${tree}/graph_tgsmall exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}_fglarge_rescored
  done
fi

if [ $stage -le 5 ]; then
  echo "$0: scoring per condition"
  for ds in $test_sets; do
    decode_dir=exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}_fglarge_rescored
    score_result=${decode_dir}/scoring_kaldi/wer_details/per_utt
    for cond in $conditions; do
      # get nerror
      nerr=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$4+$5+$6} END {print sum}'`
      # get nwords from references (NF-2 means to exclude utterance id and " ref ")
      nwrd=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$3+$4+$6} END {print sum}'`
      # compute wer with scale=2
      wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`  
      # report the results
      echo "Condition $cond: wer $wer"
    done >${decode_dir}/scoring_kaldi/best_wer_cond
    cat ${decode_dir}/scoring_kaldi/best_wer_cond
  done
fi
