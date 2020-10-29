#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
# End configuration section

data_affix= # if provided, simulated data with this affix will be used

. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <corpus-dir> <librispeech-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora/LibriCSS /export/corpora/LibriSpeech"
  exit 1
fi

corpus_dir=$1
librispeech_dir=$2

set -e -o pipefail

if ! [ -d $corpus_dir ]; then
  echo "$0: $corpus_dir does not exist. Please run the data simulation first."
  exit 1
fi

for dataset in train dev test; do
  echo "$0: Preparing $dataset data.."
  output_data_dir=data/${dataset}_sim${data_affix}
  if [ -d $output_data_dir ]; then
    echo "$0: $output_data_dir already exists. Please remove to continue."
    exit 1
  fi
  wav_data_dir=$corpus_dir/data/SimLibriCSS-${dataset}${data_affix}/
  mkdir -p ${output_data_dir}
  local/prepare_simulated_meetings_data.py --txtpath $librispeech_dir \
    --wavpath $wav_data_dir --tgtpath $output_data_dir --type $dataset
  utils/fix_data_dir.sh ${output_data_dir}

  # also fix the wav_clean.scp
  file=$output_data_dir/wav_clean.scp
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
done

exit 0