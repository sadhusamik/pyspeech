#! /bin/bash

# Upper level scripts to run several monophone ASR training and decoding 
# in a sequential manner 
#
# Samik Sadhu 


./run_ASR_monostate.sh \
  --stage 2 \
  --exp_dir exp_lrr_monostate_monophone \
  --feat_suff lrr || exit 1;

# Send an e-mail when finished 

mail -s "JOB DONE" "sadhusamik@gmail.com" <<EOF
LRR training and decoding done successfully!"
EOF

./run_ASR_monostate.sh \
  --exp_dir exp_srr_monostate_monophone \
  --feat_suff srr || exit 1;

mail -s "JOB DONE" "sadhusamik@gmail.com" <<EOF
SRR training and decoding done successfully! 
EOF
