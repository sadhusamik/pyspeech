#!/bin/bash 

. ./path.sh

# Script to train a multimodal vanilla nnet given frame level features and labels
# using pytorch and GPU

# We split the data based on the data directory size

nj=5
stage=0
cmd=queue.pl
model_name=model.nnet
ntargets=48
nlayers=4
nunits=256
bsize=10000
epochs=20
cv_stop=5
activation=tanh
remove_egs=false

. parse_options.sh || exit 1;

echo "$0 $@" 

egs_1=$1
egs_2=$2
nnet_dir=$3


# Generate data to make things faster 
egs_dir=$nnet_dir/egs

if [ $stage -le 0 ]; then 
  echo "$0: Combining training egs from" $egs_1 "and" $egs_2
  
  if [ -d $egs_dir ] ; then  rm -r $egs_dir ; fi
    for dset in train test; do 
      # Generate training egs
      mkdir -p $egs_dir/$dset
      combine_features.py $egs_1/$dset $egs_2/$dset $egs_dir/$dset \
        --egs_type=post || exit 1;
      # copy all labels 
      cp -r $egs_1/$dset/labels.*.egs $egs_dir/$dset/
    done
fi

split_num=`ls $egs_dir/train/data.*.egs | wc -l`
echo $((2*$ntargets)) > $egs_dir/dim

# Now train the network 
if [ $stage -le 1 ]; then 
  mkdir -p $nnet_dir

  echo "$0: Begin main nnet training, waiting for machine"

  $cmd JOB=1 $nnet_dir/train.log \
    train_vanilla_nnet.py \
    $egs_dir \
    $nnet_dir/model.nnet \
    --ntargets=$ntargets \
    --nlayers=$nlayers \
    --nunits=$nunits \
    --bsize=$bsize \
    --split_num=$split_num \
    --epochs=$epochs \
    --cv_stop=$cv_stop \
    --activation=$activation || exit 1;
fi

if $remove_egs ; then
  rm -r $egs_dir
fi

echo "$0: Finished training and testing vanilla nnet"
