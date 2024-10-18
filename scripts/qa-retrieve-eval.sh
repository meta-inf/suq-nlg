shopt -s nullglob

for path in $@; do
    files=($path/eval_*gpt-4o-mini*/map_.pkl)
    if [[ ${#files[@]} -ne 1 ]]; then
        echo "incorrect layout: $path"
        continue
    fi
    if compgen -G "$path/eval_*gpt-4o-mini*/map__parsed.pkl" > /dev/null; then
        echo "Skipping $path"
        continue
    fi
    (set -x; python -m qa.eval_parse -r ${files[0]})
done
