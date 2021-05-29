#!/bin/bash

# Usage: ./test_runner.sh lab6.py

python3 $1 -svd custom -k 5 -f img/pink-floyd.jpg
python3 $1 -svd library -k 5 -f img/pink-floyd.jpg
python3 $1 -svd custom -k 30 -f img/pink-floyd.jpg
python3 $1 -svd custom -k 30 -f img/wittgenstein.jpg
python3 $1 -svd custom -k 70 -f img/clouds.jpg

