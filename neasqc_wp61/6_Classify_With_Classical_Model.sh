#!/bin/bash

echo 'This script classifies examples using classical classifier model.'


while getopts t:v:j:s:r:i:o:b:l:w:z:g:y:c: flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        j) validation=${OPTARG};;
        s) seed=${OPTARG};;
        r) runs=${OPTARG};;
        i) iterations=${OPTARG};;
        o) outfile=${OPTARG};;

        b) b=${OPTARG};;
        l) lr=${OPTARG};;
        w) wd=${OPTARG};;
        z) slr=${OPTARG};;
        g) g=${OPTARG};;

        y) version=${OPTARG};;
        c) pca=${OPTARG};;
    esac
done

echo "train: $train";
echo "test: $test";
echo "validation: $validation";
echo "seed: $seed";
echo "epochs: $epochs";
echo "runs: $runs";
echo "iterations: $iterations";
echo "outfile: $outfile";
echo "Batch size: $b";
echo "Learning rate: $lr";
echo "Weight decay: $wd";
echo "Step size for the learning rate scheduler: $slr";
echo "Gamma for the learning rate scheduler: $g";
echo "Version between Classical 1, Classical 2 and Classical 3: $version";
echo "Reduced dimension for the word embeddings: $pca";


echo "Classical counterpart"
python3.10 ./data/data_processing/use_alpha_classical_counterpart.py -s ${seed} -i ${iterations} -r ${runs} -c ${version} -pca ${pca} -tr ${train} -te ${test} -val ${validation} -o ${outfile} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}