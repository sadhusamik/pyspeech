#!/bin/bash 

. ./path.sh

# LIP FEATURE  options 

nj=100
context=
dim=50
predictor_path='exp/shape_predictor_68_face_landmarks.dat'
cmd=queue.pl
ark_cmd=copy-feats

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

if [ -f $segment ]; then 
  echo $0": Splitting Segment files..."

  split_segments=""
  for n in $(seq $nj); do 
    split_segments="$split_segments $log_dir/segments.$n"
  done
 utils/split_scp.pl $segment $split_segments || exit 1;

 echo $0": Computing lip features for segment files..."

  # Compute lip features 

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    get_lip_feature.py \
      $log_dir/segments.JOB \
      $feat_dir/lipfeats_${name}.JOB \
      $predictor_path \
      --dim=$dim \
      --kaldi_cmd=$ark_cmd || exit 1

# concatenate all scp files together 

for n in $(seq $nj); do 
  cat $feat_dir/lipfeats_$name.$n.scp || exit 1;
done > $data_dir/feats.scp
 

rm $log_dir/segments.*

elif [ -f $scp ]; then
  echo "$0: Splitting scp file.."
  split_scp=""
  for n in $(seq $nj); do 
    split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scp || exit 1;
  echo "$0: Computing lip features"

  $cmd  JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    python3 ../../src/featgen/get_lip_feature.py \
      $log_dir/wav_${name}.JOB.scp \
      $feat_dir/lipfeats_${name}.JOB \
      $predictor_path \
      --dim=$dim \
      --kaldi_cmd=$ark_cmd || exit 1

  # concatenate all scp files together 

  for n in $(seq $nj); do 
    cat $feat_dir/lipfeats_$name.$n.scp || exit 1;
  done > $data_dir/feats.scp

  rm $log_dir/wav_${name}.*.scp

else
  echo $0": Neither scp file nor segment file exists... something is wrong!"
  exit 1;
fi

echo $0": Finished computed lip features for $name"
