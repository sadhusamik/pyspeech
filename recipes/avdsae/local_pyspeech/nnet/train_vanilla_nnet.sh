#!/bin/bash 

. ./path.sh

# Script to train a vanilla nnet given frame level features and labels
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

data_train=$1
data_test=$2
nnet_dir=$3

# First generate the batches of data

if [ $nj = "auto" ]; then 
  size=`du -sh data/train_v/ | cut -f1 | rev | cut -c 2- | rev`
  size=`echo "($size)/1" | bc`
  if [ $size -le 8 ]; then 
    echo "$0:  Data size is less than 8GB, not splitting data"
    split_num=1 # Just create one segent
    echo "$0: Splitting training data into" $split_num "number of mega-batches"
    split_data.sh $data_train $split_num || exit 1;
  else
    split_num=`python -c "import numpy; print(int(numpy.ceil($size/8)))"` 
    echo "$0: Splitting training data into" $split_num "number of mega-batches"
    split_data.sh $data_train $split_num || exit 1;
  fi 
else
  split_num=$nj
  echo "$0: Splitting training data into" $split_num "number of mega-batches"
  split_data.sh $data_train $split_num || exit 1;
fi



# Generate data to make things faster 
egs_dir=$nnet_dir/egs

if [ $stage -le 0 ]; then 
  echo "$0: Generating training and test examples for nnet"
  
  if [ -d $egs_dir ] ; then  rm -r $egs_dir ; fi
  mkdir -p $egs_dir
  generate_egs.py $data_train $data_test $egs_dir \
    --split_num=$split_num || exit 1;
fi

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
