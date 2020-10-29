#!/usr/bin/env bash

# Copyright     2013  Daniel Povey
#               2020  Desh Raj
# Apache 2.0.


# This script has similar functionality as local/nnet3/target_speaker/internal/extract_ivectors.sh, 
# with an additional option for choosing to get xvectors of the speaker from a 
# different utterance of the same speaker.

# Begin configuration section.
nj=30
cmd="queue.pl -l hostname='!c2*\&!b1[48]*\&!a*'"
stage=0

xvector_type=all    # can be "fixed", "shuffle" or "all". "fixed" means each utterance
                    # has a unique i-vector, "shuffle" means the i-vector for an
                    # utterance is extracted from a different utterance of the same
                    # speaker, "all" means speaker-level i-vectors.
chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet
                  # directory.
cache_capacity=64 # Cache capacity for x-vector extractor
compress=true       # If true, compress the xVectors stored on disk (it's lossy
                    # compression, as used for feature matrices)
num_threads=1 # Number of threads used by ivector-extract.  It is usually not
              # helpful to set this to > 1.  It is only useful if you have
              # fewer speakers than the number of jobs you want to run.
use_gpu=false

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <extractor-dir> <xvector-dir>"
  echo " e.g.: $0 data/test exp/nnet2_online/extractor exp/nnet2_online/ivectors_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-threads)"
  echo "  --num-threads <n|1>                              # Number of threads for each job"
  echo "                                                   # Ignored if <alignment-dir> or <decode-dir> supplied."
  echo "  --stage <stage|0>                                # To control partial reruns"
  exit 1;
fi

feats_data_dir=$1
srcdir=$2
dir=$3

for f in $srcdir/final.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $feats_data_dir/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat $srcdir/min_chunk_size 2>/dev/null`
max_chunk_size=`cat $srcdir/max_chunk_size 2>/dev/null`

nnet=$srcdir/final.raw
if [ -f $srcdir/extract.config ] ; then
  echo "$0: using $srcdir/extract.config to extract xvectors"
  nnet="nnet3-copy --nnet-config=$srcdir/extract.config $srcdir/final.raw - |"
fi

if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi

mkdir -p $dir/log

utils/split_data.sh $feats_data_dir $nj
echo "$0: extracting xvectors for $feats_data_dir"
sdata=$feats_data_dir/split$nj/JOB

# Set up the features
feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd --gpu 1 ${dir}/log/extract.$g.log \
        nnet3-xvector-compute --use-gpu=yes --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
        "$nnet" "`echo $feat | sed s/JOB/$g/g`" ark,scp:${dir}/xvector.$g.ark,${dir}/xvector.$g.scp || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet3-xvector-compute --use-gpu=no --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
      "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  if [ $xvector_type == "shuffle" ]; then
    # Shuffle x-vectors of same speaker
    echo "$0: shuffling ivectors of same speaker"
    for spk_id in $(cut -d' ' -f1 $feats_data_dir/spk2utt); do
      ids=$(grep "^${spk_id}-\|^${spk_id}_" $dir/xvector.scp | cut -d' ' -f1 )
      scps=$(grep "^${spk_id}-\|^${spk_id}_" $dir/xvector.scp | cut -d' ' -f2 | sort -R )
      paste <(echo "$ids") <(echo "$scps") -d ' '
    done >$dir/xvector_spk.scp.tmp
    sort -u $dir/xvector_spk.scp.tmp > $dir/xvectors.scp
  elif [ $xvector_type == "all" ]; then
    # Average the utterance-level xvectors to get speaker-level xvectors.
    echo "$0: computing mean of xvectors for each speaker"
    $cmd $dir/log/speaker_mean.log \
      ivector-mean ark:$feats_data_dir/spk2utt scp:$dir/xvector.scp \
      ark,scp:$dir/xvectors.ark,$dir/xvectors.scp ark,t:$dir/num_utts.ark || exit 1;
  elif [ $xvector_type == "fixed" ]; then
    # Use same utterance for x-vector computation
    mv $dir/xvector.scp $dir/xvectors.scp
  fi
fi

echo "$0: done extracting x-vectors to $dir using the extractor in $srcdir."

