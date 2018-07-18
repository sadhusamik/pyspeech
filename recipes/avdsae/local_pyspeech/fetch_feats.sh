#!/bin/bash

## Script to fetch the selected features from a data dir into feats.scp and
## cmvn.scp
## Samik Sadhu 

data_dir=$1
feat_type=$2
get_cmvn=false

. ./utils/parse_options.sh

name=`basename $data_dir`
#basefeat=`echo $feat_type | cut -f1 -d'_'`

#counter=1
#while true ; do 
#  fname=$data_dir/$feat_type/${basefeat}_$name.$counter.scp
#  if [ -f $fname ] ; then 
#    cat $fname; 
#    counter=$((counter+1))
#  else
#    break;
#  fi
#done > $data_dir/feats.scp

for x in `ls $data_dir/$feat_type/*scp`; do 
  cat $x
done > $data_dir/feats.scp

#for n in `find $data_dir/$feat_type/ -name ${basefeat}_$name'.*.scp'`; do 
#   cat $n || exit 1;
#done > $data_dir/feats.scp

#for n in $(seq $nj); do 
#  cat $data_dir/$feat_type/${basefeat}_$name.$n.scp || exit 1;
#done > $data_dir/feats.scp

if $get_cmvn; then 
  cmvn_dir=$data_dir/$feat_type/cmvn

  cp $cmvn_dir/cmvn_$name.scp $data_dir/cmvn.scp || exit 1;
fi

echo $0": Compiled all the precomputed $name features from $data_dir !"
