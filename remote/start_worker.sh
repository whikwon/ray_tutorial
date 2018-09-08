#!/bin/bash

export PATH="/home/medipixel/anaconda3/bin:$PATH"
source activate opensim-rl
ray stop
ray start --redis-address=10.0.1.8:8787
