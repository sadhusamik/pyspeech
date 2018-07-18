#!/bin/bash 

# Script to convert a kaldi generated ctm file to a bunch of PHN files


ctm_file=$1
PHN_file_dir=$2

mkdir -p $PHN_file_dir

for uttid in `cat $ctm_file | cut -d' ' -f1 | uniq`; do
  phn_file=$PHN_file_dir/$uttid.PHN 
  grep $uttid $ctm_file | cut -d' ' -f3-5 > $phn_file
done
