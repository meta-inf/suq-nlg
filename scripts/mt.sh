declare -A langs
langs["tam"]="tam_Taml"
langs["kmr"]="kmr_Latn"

PREFIX=./run/mt
mkdir -p $PREFIX/logs

function run_base {
    model=$1
    ngt=$2
    langPref=$3
    langName=${langs[$langPref]}
    shift 3
    (set -x;
     python -m mt.main -production --mode mt -lp $PREFIX/preproc-$langPref-full/ --dest_lang ${langName} -comp 0 \
       --n_epis_chains 0 --n_intra_chain_samples 16 --lm $model --n_gt_demonstrations $ngt \
       $@ > $PREFIX/logs/BASE-$model-$langPref-gt$ngt.log 2>&1
    )
}

function run_eu {
    model=$1
    ngt=$2
    langPref=$3
    langName=${langs[$langPref]}
    shift 3
    (set -x;
     python -m mt.main -production --mode mt -lp $PREFIX/preproc-$langPref-full/ --dest_lang ${langName} -comp 4 \
       --n_epis_chains 5 --n_intra_chain_samples 8 --lm $model --n_gt_demonstrations $ngt \
       $@ > $PREFIX/logs/$model-$langPref-N500-comp4-gt$ngt.log 2>&1
    )
}

for model in gpt-4o-mini gpt-4o gemini-1.5-pro; do
    for lang in kmr tam; do  
        for ngt in 128 4; do
            run_base $model $ngt $lang -dir $PREFIX/BASE-$model-$lang-gt$ngt 
        done
        run_eu $model 4 $lang -dir $PREFIX/$model-$lang-N500-comp4-gt4
    done
done