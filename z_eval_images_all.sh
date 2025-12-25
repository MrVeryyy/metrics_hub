#!/bin/bash

dig=3
env_py=$HOME"/miniconda3/envs/metrics_hub/bin/python"
${env_py} scripts/eval_pair_all.py  --dig ${dig}