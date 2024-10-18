shopt -s nullglob

# get list of paths from args
paths=""
for arg in "$@"
do
    if [ -f "$arg/records_pp.pkl" ]; then
        paths="$paths $arg"
    else
        echo "$arg/records_pp.pkl does not exist"
    fi
done
for path in $paths; do
    echo $path
done

echo "Continue?"
read -r response

for path in $paths; do
    files=($path/eval_*gpt-4o-mini*/map__submission.jsonl)
    echo "$path" $files ${#files[@]}
    if [[ ${#files[@]} -gt 0 ]]; then
        echo "Skipping $path"
        continue
    fi
    (set -x; python -m utils.openai_dump_eval -ls $path -dp RUN)
    files=($path/eval_*gpt-4o-mini*/map_.pkl)
    if [[ ${#files[@]} -ne 1 ]]; then
        echo "$path wasn't dumped: ${files}"
        continue
    fi
    (set -x; python -m qa.eval_submit -i $path/eval_*gpt-4o-mini*/requests_.jsonl --run)
    sleep 60
done
