#!/bin/bash

# Prepares the data for TIMIT AV speech recognition 


source_dir=$1
data_dir=$2
data_dir_timit=$3

# echo out the command 
echo "Running command: $0 $@"
source_dir=$source_dir/av_db_am_eng/data

# Get the list of speakers 

spk_list=`ls -d $source_dir/* | tr '\n' '\0' | xargs -0 -n 1 basename`

spk_female=`for spk in $spk_list; do  if [[ $spk == 'F'* ]]; then echo $spk; fi done`
spk_male=`for spk in $spk_list; do  if [[ $spk == 'M'* ]]; then echo $spk ; fi done`
spk_female=($spk_female); spk_male=($spk_male); # Convert them to arrays
nspk_male=${#spk_male[@]}; nspk_female=${#spk_female[@]}

echo "$0: Found" $nspk_male "male speakers :" ${spk_male[@]} 
echo "$0: Found" $nspk_female "female speakers :" ${spk_female[@]}

# Divide training and test sets 
# We keep one male and female speaker for testing 
# Remaining speakers are used for training models

mkdir -p $data_dir/train $data_dir/test

> $data_dir/test/spk_list
> $data_dir/train/spk_list

for n in $(seq 0 $(($nspk_male-2))); do 
  echo ${spk_male[$n]}; 
done >> $data_dir/train/spk_list

for n in $(seq 0 $(($nspk_female-2))); do 
  echo ${spk_female[$n]} ; 
done >>  $data_dir/train/spk_list

echo ${spk_male[((nspk_male-1))]} >> $data_dir/test/spk_list
echo ${spk_female[((nspk_female-1))]} >> $data_dir/test/spk_list

# Generate the wav.scp files for train and test set 

echo "$0: Generating list of wav files"

for dset in train test; do 
  > $data_dir/$dset/wav.scp 
  for spk in `cat $data_dir/$dset/spk_list`; do 
    for file in `ls $source_dir/$spk` ; do 
      if [[ $file == *'TIMIT'* ]]; then
        full_path=$source_dir/$spk/$file 
        fname=`basename $file`; uttid=`echo $fname | cut -d'.' -f1`
        echo "$uttid ffmpeg -loglevel panic -i $full_path -f wav -ar 16000 -ac 1 - |"
      fi
    done
  done >> $data_dir/$dset/wav.scp 
done

# Generate utt2spk

echo "$0: Generating utt2spk files"

for dset in train test ; do 
  > $data_dir/$dset/utt2spk
  for spk in `cat $data_dir/$dset/spk_list`; do 
    for file in `ls $source_dir/$spk` ; do 
      if [[ $file == *'TIMIT'* ]]; then
        fname=`basename $file`; uttid=`echo $fname | cut -d'.' -f1`
        echo "$uttid $spk"
      fi
    done
  done >> $data_dir/$dset/utt2spk 
done

# Generate spk2utt file 

echo "$0: Generating spk2utt files"

for dset in train test; do 
  utils/utt2spk_to_spk2utt.pl $data_dir/$dset/utt2spk > $data_dir/$dset/spk2utt || exit 1;
done

# Generate the text files 

echo "$0: Generating text files"

if [ ! -d $data_dir_timit/train ] || [ ! -d $data_dir_timit/test ] || [ ! -d $data_dir_timit/dev ] || [ ! -d $data_dir_timit/lang ]; then 
  echo "$0: TIMIT data directory is not properly structured!"
  exit 1;
fi

for dset in train test ; do 
  > $data_dir/$dset/text
  for spk in `cat $data_dir/$dset/spk_list`; do 
    for file in `ls $source_dir/$spk` ; do 
      if [[ $file == *'TIMIT'* ]]; then
        fname=`basename $file`
        uttid=`echo $fname | cut -d'.' -f1`
        main_uttid=`echo $uttid | cut -d'_' -f3`
        echo $uttid `for x in train test dev ; do \
          cat $data_dir_timit/$x/text | grep $main_uttid; done |\
          head -n 1 | cut -d' ' -f2-`
      fi
    done
  done >> $data_dir/$dset/text
done


