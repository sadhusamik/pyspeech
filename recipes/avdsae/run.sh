#!/bin/bash 

# Script to run multi-modal-machines on Audiovisual Database of 
# Spoken American English

stage=3
data_dir=data
data_dir_timit=data_timit
exp_dir=exp
mfccdir_timit=mfcc_timit
mfccdir_avdsae=mfcc_avdsae
lipfeat_dir=lipfeats
PHN_file_dir=$data_dir/PHN_files
normal_dir=normalized

kaldi_mfcc=false  # If true we use kaldi for computing mfcc features

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh 

avdsae='/export/corpora/LDC/LDC2009V01'
timit='/export/corpora5/LDC/LDC93S1/timit/TIMIT'


# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=5

printf '\nDATA PREPARATION FOR TIMIT\n\n'

if [ $stage -le 0 ]; then 

  # First change the speaker list to 100 for dev and 68 for test
 
  ls -d $timit/TEST/DR*/* | rev | cut -d'/' -f1 |\
    uniq | tail -n 68 | tr '[:upper:]' '[:lower:]' | rev > conf/test_spk.list
  ls -d $timit/TEST/DR*/* | rev | cut -d'/' -f1 |\
    uniq | head -n 100 | tr '[:upper:]' '[:lower:]' | rev > conf/dev_spk.list

  local/timit_data_prep.sh $timit || exit 1

  local/timit_prepare_dict.sh

  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
   data/local/dict "sil" data/local/lang_tmp data/lang

  local/timit_format_data.sh

  mv data $data_dir_timit # Move to a seperate directory for timit stuff
fi

printf '\nDONE\n'


printf 'DATA PREPARATION FOR AVDSAE DATABASE\n\n'

if [ $stage -le 1 ]; then 
 ./local_pyspeech/prepare_timit_avdsae.sh \
   $avdsae \
   $data_dir \
   $data_dir_timit || exit 1;
  
  #copy rest of the important things from TIMIT data directory 
  cp -r $data_dir_timit/local $data_dir_timit/lang $data_dir/lang_test_bg \
    data_dir || exit 1;
  
fi

printf '\nDONE\n\n'

printf '\nCOMPUTE MFCC FEATURES FOR TIMIT\n\n'

if [ $stage -le 2 ]; then 

for x in train test; do
  if $kaldi_mfcc ; then
    steps/make_mfcc.sh --cmd "$train_cmd" \
      --nj $feats_nj \
      $data_dir_timit/$x \
      $exp_dir/make_mfcc/$x \
      $mfccdir_timit
    
    steps/compute_cmvn_stats.sh $data_dir_timit/$x \
      $exp_dir/make_mfcc/$x \
      $mfccdir_timit
  else
    add_opts=  # additional options for mfcc computation
    local_pyspeech/make_mfcc_feats.sh --nj $feats_nj \
      $data_dir_timit/$x \
      $data_dir_timit/$x/$mfccdir_timit $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      $data_dir_timit/$x \
      $data_dir_timit/$x/$mfccdir_timit/cmvn || exit 1;
  fi
done
fi

printf '\nCOMPUTE MFCC FEATURES FOR AVDSAE\n\n'

if [ $stage -le 3 ]; then 

for x in train test; do
  if $kaldi_mfcc ; then
    steps/make_mfcc.sh --cmd "$train_cmd" \
      --nj $feats_nj \
      $data_dir/$x \
      $exp_dir/make_mfcc/$x \
      $mfccdir_avdsae || exit 1;
    
    steps/compute_cmvn_stats.sh $data_dir/$x \
      $exp_dir/make_mfcc/$x \
      $mfccdir_avdsae || exit 1;
  else
    add_opts=  # additional options for mfcc computation
    local_pyspeech/make_mfcc_feats.sh --nj $feats_nj \
      $data_dir/$x \
      $data_dir/$x/$mfccdir_avdsae $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      $data_dir/$x \
      $data_dir/$x/$mfccdir_avdsae/cmvn || exit 1;
  fi
done
fi
printf '\nDONE\n'

printf '\nTRAIN MONOPHONE HMM MODEL ON TIMIT\n\n'

if [ $stage -le 4 ]; then

  steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" \
    $data_dir_timit/train \
    $data_dir_timit/lang \
    $exp_dir/mono || exit 1;

  utils/mkgraph.sh $data_dir_timit/lang_test_bg \
    $exp_dir/mono \
    $exp_dir/mono/graph || exit 1;

  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
   $exp_dir/mono/graph \
   $data_dir_timit/test \
   $exp_dir/mono/decode_test || exit 1;
fi

printf '\nDONE\n'

printf '\nFORCED ALIGNMENT ON AVDSAE DATA\n\n'

if [ $stage -le 5 ]; then 
  for dset in train ; do
    steps/align_si.sh --nj 10 \
      --cmd "$train_cmd" \
      $data_dir/$dset \
      $data_dir/lang \
      $exp_dir/mono \
      $exp_dir/mono_avdsae_${dset}_ali || exit 1;
  done

  for dset in test ; do
    steps/align_si.sh --nj 2 \
      --cmd "$train_cmd" \
      $data_dir/$dset \
      $data_dir/lang \
      $exp_dir/mono \
      $exp_dir/mono_avdsae_${dset}_ali || exit 1;
  done

  # Generate the ctm files 
  for dset in train test; do 
    steps/get_train_ctm.sh $data_dir/$dset \
      $data_dir/lang \
      $exp_dir/mono_avdsae_${dset}_ali || exit 1;
  done
fi

printf '\nDONE\n'

printf '\nGENERATE PHONE FILES FROM ALIGNMENT FILES FOR AVDSAE\n\n'

if [ $stage -le 6 ]; then 
 for dset in train test; do 
  utils_pyspeech/ctm2phn.sh exp/mono_avdsae_${dset}_ali/ctm \
    $PHN_file_dir || exit 1;
 done
fi

printf '\nDONE\n'

printf '\nPREPARE VIDEO DATA\n\n'

if [ $stage -le 7 ]; then 
 for dset in train test; do 
  local_pyspeech/prepare_video_data.sh $avdsae $data_dir || exit 1;
 done 
fi

printf '\nDONE\n'

printf '\nCOMPUTE LIP FEATURES\n\n'
if [ $stage -le 8 ]; then
  for x in train test; do 
    add_opts=
    local_pyspeech/make_lip_feats.sh --nj 50 --cmd "$train_cmd" \
      $data_dir/${x}_v \
      $data_dir/${x}_v/$lipfeat_dir $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      $data_dir/${x}_v \
      $data_dir/${x}_v/$lipfeat_dir/cmvn || exit 1;

  done
fi
printf '\nDONE\n'


printf '\nORGANIZE MULTIMODAL DATA FOR NNET TRAINING\n\n'

if [ $stage -le 9 ]; then
  for dset in train test; do
    organize_multimodal_data.py \
      $data_dir/${dset} $data_dir/${dset}_v \
      --split_num=10 || exit 1;
  done
fi

printf '\nDONE\n'

printf '\nOBTAIN FRAME-WISE PHONEME LABELS\n\n'

if [ $stage -le 10 ]; then 
  for dset in train test; do 
    
    # Obtain phoneme labels
    PHN_file_2_frame_labels.py $data_dir/$dset \
     $PHN_file_dir \
     ./conf/phones.txt || exit 1;

    ./local_pyspeech/fetch_feats.sh $data_dir/$dset normalized || exit 1; 
     
    # Check if frame numbers are same for labels and features 
    normalize_labels.py $data_dir/$dset || exit 1; 
  done
fi

printf '\nDONE\n'

nnet_jobs=2

printf '\nTRAIN AN NNET WITH ONLY ACOUSTIC DATA\n\n'

if [ $stage -le 11 ]; then
 ./local_pyspeech/nnet/train_vanilla_nnet.sh \
   --cmd "$cuda_cmd" \
   --nj $nnet_jobs \
   --stage 1 \
   --nlayers 5 \
   --nunits 512 \
   --bsize 2000 \
   $data_dir/train \
   $data_dir/test \
   $exp_dir/nnet_acoustic ||  exit 1; 
fi

printf '\nDONE\n'
 
printf '\nTRAIN A VANILLA NNET WITH ONLY VISUAL DATA\n\n'

if [ $stage -le 12 ]; then
  
  for dset in train test; do
    local_pyspeech/fetch_feats.sh $data_dir/${dset}_v "normalized" || exit 1;
    local_pyspeech/get_cmvn.sh \
      $data_dir/${dset}_v \
      $data_dir/${dset}_v/normalized/cmvn || exit 1;
  done

  # Train the net
  ./local_pyspeech/nnet/train_vanilla_nnet.sh \
   --cmd "$cuda_cmd" \
   --nj $nnet_jobs \
   --stage 0 \
   --nlayers 5 \
   --nunits 512 \
   --bsize 20000 \
   $data_dir/train_v \
   $data_dir/test_v \
   $exp_dir/nnet_visual ||  exit 1; 
fi

printf '\nDONE\n'


printf '\nTRAIN A CNN NNET WITH VISUAL FEATURES\n\n'

if  [ $stage -le 13 ]; then 

   local_pyspeech/nnet/train_cnn_nnet.sh \
   --cmd "$cuda_cmd" \
   --nj $nnet_jobs \
   --stage 1 \
   --nlayers 2 \
   --ndepth 20 \
   --bsize 1000 \
   --weight_decay .00001 \
   $data_dir/train_v \
   $data_dir/test_v \
   $exp_dir/nnet_visual_cnn ||  exit 1; 
fi

printf '\nDONE\n'

printf '\nGENERATE POSTERIORS FROM THE NNET\n\n'

if [ $stage -le 14 ]; then 
  for exp in nnet_acoustic nnet_visual; do   
    for dset in train test; do
     mkdir -p $exp_dir/$exp/post/$dset 
     egs_2_post.py $exp_dir/$exp/model.nnet \
       $exp_dir/$exp/egs/$dset \
       $exp_dir/$exp/post/$dset || exit 1;
     cp -r $exp_dir/$exp/egs/$dset/labels.*.egs $exp_dir/$exp/post/$dset/ 
    done
  done
fi
printf '\nDONE\n'

printf '\n'

printf '\nTRAIN MULTIMODAL NNET\n\n'

if [ $stage -le 15 ]; then 
  local_pyspeech/nnet/train_multimodal.sh \
    --cmd "$cuda_cmd" \
    --nj $nnet_jobs \
    --stage 1 \
    --nlayers 5 \
    --nunits 512 \
    --bsize 2000 \
    $exp_dir/nnet_acoustic/post \
    $exp_dir/nnet_visual/post \
    $exp_dir/nnet_multimodal || exit 1;

fi

printf '\nDONE\n'

bash RESULTS $exp_dir

echo "Experiment completed successfully on " `date`
