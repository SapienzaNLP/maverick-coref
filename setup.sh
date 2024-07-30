#!/bin/bash
conda activate maverick-coref

cd data/prepare_ontonotes
conda create -y --name py27 python=2.7

eval "$(conda shell.bash hook)"

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz
dlx $conll_url conll-2012-scripts.v3.tar.gz

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer

tar -xzvf ontonotes-release-5.0_LDC2013T19.tgz
ontonotes_path=ontonotes-release-5.0

conda activate py27
bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

function compile_partition() {
    rm -f $2.$5.$3$4
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> $2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _gold_conll $1
    compile_partition train train v4 _gold_conll $1
    compile_partition test test v4 _gold_conll $1
}

compile_language english

conda activate maverick-coref
python minimize.py
python prepare_tests_files.py

