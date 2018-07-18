export PYSPEECH_ROOT=$PWD/../../
export PYSPEECH_TOOLS=$PYSPEECH_ROOT/src
export PATH=$PATH:$PYSPEECH_TOOLS
export KALDI_ROOT=$PYSPEECH_ROOT/tools/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=$PYSPEECH_TOOLS/utils:$PYSPEECH_TOOLS/featgen/:$PYSPEECH_TOOLS/nnet/:$PATH
