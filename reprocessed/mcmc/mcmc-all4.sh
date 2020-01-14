#!/bin/bash
#$ -pe smp-verbose 1 -cwd
module add apps/python/2.7.8/withtk
module add apps/setuptools/15.1/python-2.7.8
module add compilers/gcc/5.1.0
python MCMC-all4.py
