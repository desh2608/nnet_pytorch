#!/usr/bin/env bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.


# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B


if [ $stage -le 0 ]; then
  # data preparation. 
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;
  
  # Prepare dictionary
  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  (
    local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1  && \
      utils/prepare_lang.sh data/local/dict_nosp_larger \
                            "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
      local/wsj_train_lms.sh --dict-suffix "_nosp" &&
      local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  ) &
fi

if [ $stage -le 1 ]; then 
  # MFCC extraction for training bootstrapping GMM models
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/train_si284 || exit 1;
  steps/compute_cmvn_stats.sh data/train_si284 || exit 1;
fi

if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

  # Now make subset with the shortest 2k utterances from si-84.
  utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

  # Now make subset with half of the data from si-84.
  utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;
fi

if [ $stage -le 3 ]; then
  # monophone
  # Note: the --boost-silence option should probably be omitted by default
  # for normal setups.  It doesn't always help. [it's to discourage non-silence
  # models from modeling silence.]
  steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train_si84_2kshort data/lang_nosp exp/mono0a || exit 1;
fi

if [ $stage -le 4 ]; then
  # tri1
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train_si84_half data/lang_nosp exp/mono0a exp/mono0a_ali || exit 1;

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
    data/train_si84_half data/lang_nosp exp/mono0a_ali exp/tri1 || exit 1;

fi

if [ $stage -le 5 ]; then
  # tri2b.  there is no special meaning in the "b"-- it's historical.
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train_si84 data/lang_nosp exp/tri1 exp/tri1_ali_si84 || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_si84 data/lang_nosp exp/tri1_ali_si84 exp/tri2b || exit 1;
fi

if [ $stage -le 6 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.

  # Align tri2b system with all the si284 data.
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp exp/tri2b exp/tri2b_ali_si284  || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_si284 data/lang_nosp exp/tri2b_ali_si284 exp/tri3b || exit 1;
fi

if [ $stage -le 7 ]; then
  # Estimate pronunciation and silence probabilities.

  # Silprob for normal lexicon.
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp exp/tri3b || exit 1;
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict || exit 1

  utils/prepare_lang.sh data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

  for lm_suffix in bg bg_5k tg tg_5k tgpr tgpr_5k; do
    mkdir -p data/lang_test_${lm_suffix}
    cp -r data/lang/* data/lang_test_${lm_suffix}/ || exit 1;
    rm -rf data/lang_test_${lm_suffix}/tmp
    cp data/lang_nosp_test_${lm_suffix}/G.* data/lang_test_${lm_suffix}/
  done

  # Silprob for larger ("bd") lexicon.
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp_larger \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict_larger || exit 1

  utils/prepare_lang.sh data/local/dict_larger \
    "<SPOKEN_NOISE>" data/local/lang_tmp_larger data/lang_bd || exit 1;

  for lm_suffix in tgpr tgconst tg fgpr fgconst fg; do
    mkdir -p data/lang_test_bd_${lm_suffix}
    cp -r data/lang_bd/* data/lang_test_bd_${lm_suffix}/ || exit 1;
    rm -rf data/lang_test_bd_${lm_suffix}/tmp
    cp data/lang_nosp_test_bd_${lm_suffix}/G.* data/lang_test_bd_${lm_suffix}/
  done
fi

if [ $stage -le 8 ]; then
  # From 3b system, now using data/lang as the lang directory (we have now added
  # pronunciation and silence probabilities), train another SAT system (tri4b).
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train_si284 data/lang exp/tri3b exp/tri4b || exit 1;

  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284 || exit 1;
fi
