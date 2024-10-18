# This script samples responses for the QA experiments.
# To use it: 
# - set up API access following the README
# - for any LM listed below, uncomment the respective lines and modify the dataset you want to evaluate on
# - for the pretrained Llama models, 
#     - set up Google Cloud access and deploy to an API endpoint following this link:
#       https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama3_1
#     - pass `gcp-${ENDPOINT_ID}` as $model below and modify $region accordingly
#     - clean up afterwards

nspd=10
n=1000
PREFIX=$(pwd)/run/qa
mkdir -p $PREFIX/logs

function run_for_model {
    model=$1
    region=$2
    shift 2
    for data in $@; do
        echo $data
        suff="$data-$n-$nspd-$model"
        dpath=$PREFIX/$suff
        (set -x; 
         python  -m generation_new.main -d $data -nspd $nspd -n $n -gm $model -o $dpath --gcp_region $region --run > $PREFIX/logs/${suff}.log 2>&1)
    done
}

# for model in llama3.1-8b llama3.1-70b gpt-4o-mini gpt-4o gemini-1.5-pro gemini-1.5-flash; do
#   run_for_model $model us-central1 nq_open trivia coqa sciq truthful_qa
# done
# 
# run_for_model claude-3.5 us-east5 nq_open trivia coqa sciq truthful_qa
