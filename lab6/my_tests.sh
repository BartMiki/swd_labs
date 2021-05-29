#!/bin/bash

# Usage: ./test_runner.sh lab6.py

python3 $1 -svd custom -k 5 -f img/pink-floyd.jpg -out img_out/pf_5_cst.jpg
python3 $1 -svd library -k 5 -f img/pink-floyd.jpg -out img_out/pf_5_lib.jpg

python3 $1 -svd custom -k 30 -f img/pink-floyd.jpg -out img_out/pf_30_cst.jpg
python3 $1 -svd library -k 30 -f img/pink-floyd.jpg -out img_out/pf_30_lib.jpg

python3 $1 -svd custom -k 30 -f img/wittgenstein.jpg -out img_out/wittgenstein_30_cst.jpg
python3 $1 -svd library -k 30 -f img/wittgenstein.jpg -out img_out/wittgenstein_30_lib.jpg

python3 $1 -svd custom -k 70 -f img/clouds.jpg -out img_out/clouds_70_cst.jpg
python3 $1 -svd library -k 70 -f img/clouds.jpg -out img_out/clouds_70_lib.jpg