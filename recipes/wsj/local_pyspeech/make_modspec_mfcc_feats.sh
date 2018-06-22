#!/bin/bash 
 
. ./path.sh


# MFCC options 

nj=80
nfilters=15
nmodulations=15
context=
nfft=1024
order=50
fduration_mfcc=0.02
fduration_modspec=0.5 
frate=100
cmd=queue.pl
ark_cmd=$KALDI_ROOT/src/featbin/copy-feats

. parse_options.sh || exit 1;

data_dir=$1
feat_dir=$2
add_opts=$3

# Convert feat_dir to the absolute file name 

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`
scp=$data_dir/wav.scp
segment=$data_dir/segment
log_dir=$data_dir/log
mkdir -p $log_dir

# split files

echo $0": Splitting segment OR scp files for parallalization..."

if [ -f $segment ]; then 
  echo $0": Splitting Segment files..."

  split_segments=""
  for n in $(seq $nj); do 
    split_segments="$split_segments $log_dir/segments.$n"
  done
 utils/split_scp.pl $segment $split_segments || exit 1;

 echo $0": Computing Modulation Spectral + mfcc features for segment files..."

  # Compute modspec + mfcc features 

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    computeModSpecMfccFeature.py \
      $log_dir/segments.JOB \
      $feat_dir/modspecMfcc_${name}.JOB \
      $add_opts \
      --nfilters=$nfilters \
      --nmodulations=$nmodulations \
      --order=$order \
      --nfft=$nfft \
      --fduration_mfcc=$fduration_mfcc \
      --fduration_modspec=$fduration_modspec \
      --frate=$frate \
      --kaldi_cmd=$ark_cmd || exit 1

# concatenate all scp files together 

for n in $(seq $nj); do 
  cat $feat_dir/modspecMfcc_$name.$n.scp || exit 1;
done > $data_dir/feats.scp
 

rm $log_dir/segments.*

elif [ -f $scp ]; then
  split_scp=""
  for n in $(seq $nj); do 
    split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scp || exit 1;

 echo $0": Computing Modulation Spectral + mfcc features for scp files..."

  # Compute modspec + mfcc features 

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    computeModSpecMfccFeature.py \
      $log_dir/wav_${name}.JOB.scp \
      $feat_dir/modspecMfcc_${name}.JOB \
      $add_opts \
      --nfilters=$nfilters \
      --nmodulations=$nmodulations \
      --order=$order \
      --nfft=$nfft \
      --fduration_mfcc=$fduration_mfcc \
      --fduration_modspec=$fduration_modspec \
      --frate=$frate \
      --kaldi_cmd=$ark_cmd || exit 1

  
  for n in $(seq $nj); do 
    cat $feat_dir/modspecMfcc_$name.$n.scp || exit 1;
  done > $data_dir/feats.scp

  rm $log_dir/wav_${name}.*.scp

else
  echo $0": Neither scp file nor segment file exists... something is wrong!"
  exit 1;
fi

echo $0": Finished computed mfcc features for $name"
