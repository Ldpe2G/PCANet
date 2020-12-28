[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)

PCANet
======

rewriting the PCANet matlab version into:

- c++ 
  - using openmp to reduce the run time
  - using opencv 2.x
- scala

[PCANet: A Simple Deep Learning Baseline for Image Classification?](http://arxiv.org/pdf/1404.3606v1.pdf)

examples:

use 7 persons of YaleB.

for each person, use 40 images for training, the rest for testing.
