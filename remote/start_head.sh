#!/bin/bash
export PATH="/home/medipixel/anaconda3/bin:$PATH"
source activate opensim-rl
ray stop
ray start --head --redis-port=8787
