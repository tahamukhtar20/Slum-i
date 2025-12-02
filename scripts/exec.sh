#!/bin/bash
set -e

method="$1"
config="$2"
labeled_id_path="$3"
unlabeled_id_path="$4"
save_path="$5"
now="$6"
split="$7"

echo "Running exec.sh with:"
echo "method=$method"
echo "config=$config"
echo "labeled_id_path=$labeled_id_path"
echo "unlabeled_id_path=$unlabeled_id_path"
echo "save_path=$save_path"
echo "now=$now"
echo "split=$split"

pip install -r requirements.txt

if [ "$method" == "supervised" ]; then
    python -u "${method}.py" \
        --config="$config" \
        --labeled-id-path "$labeled_id_path" \
        --save-path "$save_path" \
        2>&1 | tee "$save_path/${now}.log"
else
    python -u "${method}.py" \
        --config="$config" \
        --labeled-id-path "$labeled_id_path" \
        --unlabeled-id-path "$unlabeled_id_path" \
        --save-path "$save_path" \
        --split "$split" \
        2>&1 | tee "$save_path/${now}.log"
fi
