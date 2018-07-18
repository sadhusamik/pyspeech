#!/bin/bash

# Script to prepare the video modality data from AVDSAE

source_dir=$1
data_dir=$2

source_dir=$source_dir/av_db_am_eng/data

#echo out command 
echo "Running command: $0 $@"

# Get speaker list for train and test

mkdir -p $data_dir/train_v $data_dir/test_v

# Generate wav.scp files

echo "$0: Generating list of mov files"

for dset in train test; do 
  > $data_dir/${dset}_v/wav.scp 
  for spk in `cat $data_dir/$dset/spk_list`; do 
    for file in `ls $source_dir/$spk` ; do 
      if [[ $file == *'TIMIT'* ]]; then
        full_path=$source_dir/$spk/$file 
        fname=`basename $file`; uttid=`echo $fname | cut -d'.' -f1`
        echo "$uttid $full_path"
      fi
    done
  done >> $data_dir/${dset}_v/wav.scp 
done

# Copy rest of the important stuff 

for dset in train test; do  
  cp $data_dir/$dset/{utt2spk,spk2utt,text,spk_list} $data_dir/${dset}_v
done
