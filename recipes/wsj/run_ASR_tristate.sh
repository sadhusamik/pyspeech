#!/bin/bash

## Pyspeech & Kaldi based script to compare performance of MFCC and MBMS feature based
## ASR

## Samik Sadhu 

stage=7
train=false   # set to false to disable the training-related scripts
             # note: you probably only want to set --train false if you
             # are using at least --stage 1.
decode=true  # set to false to disable the decoding-related scripts.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

# Location of Data

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# Include all the necessary paths
. ./path.sh 

# Experiment and data directories 

exp_dir=exp_clean_tristate_1sec
data_dir=data

feat_suff=1sec   # Suffix for all feature files (Necesary of dealing with different
             # versions of the same feature)

# MFCC directory name 
mfcc=mfcc
if [ ! -z $feat_suff ]; then 
  mfcc="${mfcc}_${feat_suff}"
fi 

# MBMS directory name
modspec=modspec
if [ ! -z $feat_suff ]; then 
  modspec="${modspec}_${feat_suff}"
fi 

# MFCC-MBMS directory name 
modspecMfcc=modspecMfcc
if [ ! -z $feat_suff ]; then 
  modspecMfcc="${modspecMfcc}_${feat_suff}"
fi 

# Number of feature generation jobs
nj_mfcc=25 
nj_modspec=50

echo "DATA PREPARATION"

if [ $stage -le 0 ]; then
  # data preparation.
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

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

echo "DONE"
printf "\n"


echo "GENERATE MFCC FEATURES"

if [ $stage -le 1 ]; then
  #add_opts='--add_reverb=small_room' 
  for x in test_eval92  train_si284; do
    local_pyspeech/make_mfcc_feats.sh --nj $nj_mfcc \
      data/$x data/$x/$mfcc $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$mfcc/cmvn || exit 1;

  done

  utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

  # Now make subset with the shortest 2k utterances from si-84.
  utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

  # Now make subset with half of the data from si-84.
  utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;
fi
echo "DONE"
printf "\n"

#add_opts='--add_reverb=small_room --set_unity_gain'

echo "GENERATE MODULATION SPECTRAL FEATURES"

if [ $stage -le 2 ]; then
  
  for x in test_eval92  train_si284; do
    local_pyspeech/make_modspec_feats.sh \
      --nj $nj_modspec \
      --fduration 1 \
      --order 100 \
      data/$x data/$x/$modspec $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$modspec/cmvn || exit 1;

  done
fi
echo "DONE"
printf "\n"

#add_opts='--add_reverb=small_room'

<<skip
echo "GENERATE MODULATION SPECTRAL + MFCC COMBINED FEATURES"

if [ $stage -le 3 ]; then
  
  for x in test_eval92 train_si284; do
    local_pyspeech/make_modspec_mfcc_feats.sh \
      --nj $nj_modspec \
      --fduration_modspec 1 \
      --order 100 \
      data/$x data/$x/$modspecMfcc $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$modspecMfcc/cmvn || exit 1;

  done
fi
echo "DONE"
printf "\n"


skip
echo "MONOPHONE TRAINING"
if [ $stage -le 4 ]; then
  
  # Make sure we have the correct topo file !
  cp ./conf/topo_multi_state_hmm ./data/lang_nosp/topo

  # Fetch the mfcc features for getting alignments
  for x in train_si284 test_eval92; do
    local_pyspeech/fetch_feats.sh data/$x $mfcc $nj_mfcc
  done

  if $train; then
    steps/train_mono.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
      data/train_si284 data/lang_nosp $exp_dir/mono0a || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr $exp_dir/mono0a $exp_dir/mono0a/graph_nosp_tgpr && \
      steps/decode.sh --nj 8 --cmd "$decode_cmd" $exp_dir/mono0a/graph_nosp_tgpr \
        data/test_eval92 $exp_dir/mono0a/decode_nosp_tgpr_eval92
  fi
fi
echo "DONE"
printf "\n"

echo "TRIPHONE TRAINING WITH DELTA FEATURES"
if [ $stage -le 5 ]; then
  # tri1

  # Make sure we have the correct topo file! 
  cp ./conf/topo_multi_state_hmm ./data/lang_nosp/topo

  if $train; then
    steps/align_si.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
      data/train_si284 data/lang_nosp $exp_dir/mono0a $exp_dir/mono0a_ali || exit 1;

    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
      data/train_si284 data/lang_nosp $exp_dir/mono0a_ali $exp_dir/tri1 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr \
      $exp_dir/tri1 $exp_dir/tri1/graph_nosp_tgpr || exit 1;

    for data in eval92; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" $exp_dir/tri1/graph_nosp_tgpr \
        data/test_${data} $exp_dir/tri1/decode_nosp_tgpr_${data} || exit 1;

    done

  fi
fi
echo "DONE"
printf "\n"

<<a
echo "HYBRID ASR TRAINING WITH MFCC+MODULATION SPECTRAL FEATURES"

if [ $stage -le 6 ]; then

 if $train; then

  # Fetch the mfcc features for getting proper alignments
    for x in train_si284; do 
      local_pyspeech/fetch_feats.sh data/$x $mfcc || exit 1;
    done

  steps/align_si.sh --nj 80 --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp $exp_dir/tri1 $exp_dir/tri1_ali || exit 1;


  # Fetch the modspec features for getting hybrid model
    for x in train_si284; do 
      local_pyspeech/fetch_feats.sh data/$x $modspecMfcc || exit 1;
    done

    # DNN hybrid system training parameters
    dnn_mem_reqs="--mem 1G"
    dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

    # Run Hybrid training with modulation spectral features

    steps/nnet2/train_tanh.sh \
      --mix-up 5000 \
      --initial-learning-rate 0.015 \
      --final-learning-rate 0.002 \
      --num-hidden-layers 5  \
      --num_threads 1 \
      --parallel_opts "--gpu 1"\
      --hidden_layer_dim 256 \
      --num-jobs-nnet 16 \
      --splice_width 0 \
      --feat_type "raw" \
      --cmd "$train_cmd" \
      "${dnn_train_extra_opts[@]}" \
      data/train_si284 data/lang_nosp $exp_dir/tri1_ali $exp_dir/hybrid_modspecMfcc
  fi
  
  if $decode; then

    for x in test_eval92; do 
      local_pyspeech/fetch_feats.sh data/$x $modspecMfcc || exit 1;
    done

    [ ! -d $exp_dir/hybrid_modspecMfcc/decode_nosp_tgpr_eval92 ] \
      && mkdir -p $exp_dir/hybrid_modspecMfcc/decode_nosp_tgpr_eval92

    decode_extra_opts=(--num-threads 6)
    
    steps/nnet2/decode.sh --cmd "$decode_cmd" \
      --nj 8 \
      "${decode_extra_opts[@]}" \
      $exp_dir/tri1/graph_nosp_tgpr \
      data/test_eval92 \
      $exp_dir/hybrid_modspecMfcc/decode_nosp_tgpr_eval92 | \
      tee $exp_dir/hybrid_modspecMfcc/decode_nosp_tgpr_eval92/decode.log
  fi
fi

echo "DONE"
printf "\n"
a

echo "HYBRID ASR TRAINING WITH MFCC AND MODULATION SPECTRAL FEATURES"

if [ $stage -le 7 ]; then

 # Make sure that we have the correct topo file again 
 cp ./conf/topo_multi_state_hmm ./data/lang_nosp/topo

 if $train; then
  <<skipali  
  # Fetch the mfcc features for getting proper alignments
    for x in train_si284; do 
      local_pyspeech/fetch_feats.sh data/$x $mfcc || exit 1;
    done

  steps/align_si.sh --nj 50 --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp $exp_dir/tri1 $exp_dir/tri1_ali || exit 1;
skipali

  # Fetch the modspec features for getting hybrid model
    for x in train_si284; do 
      local_pyspeech/fetch_feats.sh data/$x $modspec || exit 1;
    done

    # DNN hybrid system training parameters
    dnn_mem_reqs="--mem 1G"
    dnn_extra_opts="--num_epochs 5 --num-epochs-extra 2 --add-layers-period 1 --shrink-interval 3"

    # Run Hybrid training with modulation spectral features

    steps/nnet2/train_tanh_fast.sh \
      --stage 196 \
      --mix-up 5000 \
      --initial-learning-rate 0.015 \
      --final-learning-rate 0.002 \
      --num-hidden-layers 5  \
      --num_threads 1 \
      --parallel_opts "-l 'hostname=hostname=b1[12345678]*|c*' --gpu 1"\
      --hidden_layer_dim 256 \
      --num-jobs-nnet 16 \
      --splice_width 0 \
      --feat_type "raw" \
      --cmd "$train_cmd" \
      "${dnn_train_extra_opts[@]}" \
      data/train_si284 data/lang_nosp \
      $exp_dir/tri1_ali $exp_dir/hybrid_modspec || exit 1;
  fi

  if $decode; then

    # Fetch the modspec features for getting hybrid model
    for x in test_eval92; do 
      local_pyspeech/fetch_feats.sh data/$x $modspec || exit 1;
    done

    [ ! -d $exp_dir/hybrid_modspec/decode_nosp_tgpr_eval92 ] \
      && mkdir -p $exp_dir/hybrid_modspec/decode_nosp_tgpr_eval92

    decode_extra_opts=(--num-threads 6)
    
    steps/nnet2/decode.sh --cmd "$decode_cmd" \
      --nj 8 \
      "${decode_extra_opts[@]}" \
      $exp_dir/tri1/graph_nosp_tgpr \
      data/test_eval92 \
      $exp_dir/hybrid_modspec/decode_nosp_tgpr_eval92 | \
      tee $exp_dir/hybrid_modspec/decode_nosp_tgpr_eval92/decode.log
  fi
<<skip3 
  # Run Hybrid training for mfcc features
  
  if $train; then

    # Fetch the mfcc features for getting hybrid model
    for x in train_si284; do 
      local_pyspeech/fetch_feats.sh data/$x $mfcc || exit 1;
    done
  
    steps/nnet2/train_tanh.sh \
      --mix-up 5000 \
      --initial-learning-rate 0.015 \
      --final-learning-rate 0.002 \
      --num-hidden-layers 5  \
      --num_threads 1 \
      --parallel_opts "--gpu 1"\
      --hidden_layer_dim 256 \
      --num-jobs-nnet 16 \
      --splice_width 15 \
      --feat_type "raw" \
      --cmd "$train_cmd" \
      "${dnn_train_extra_opts[@]}" \
      data/train_si284 data/lang_nosp $exp_dir/tri1_ali $exp_dir/hybrid_mfcc
  fi

  if $decode; then

    # Fetch the mfcc features for getting hybrid model
    for x in test_eval92 ; do 
      local_pyspeech/fetch_feats.sh data/$x $mfcc || exit 1;
    done
   
    [ ! -d $exp_dir/hybrid_mfcc/decode_nosp_tgpr_eval92 ] \
      && mkdir -p $exp_dir/hybrid_mfcc/decode_nosp_tgpr_eval92

    decode_extra_opts=(--num-threads 6)
    
    steps/nnet2/decode.sh --cmd "$decode_cmd" \
      --nj 8 \
      "${decode_extra_opts[@]}" \
      $exp_dir/tri1/graph_nosp_tgpr \
      data/test_eval92 \
      $exp_dir/hybrid_mfcc/decode_nosp_tgpr_eval92 | \
      tee $exp_dir/hybrid_mfcc/decode_nosp_tgpr_eval92/decode.log
  fi
skip3

fi

echo "DONE"
printf "\n"


echo "Experiment completed successfully on " `date`
