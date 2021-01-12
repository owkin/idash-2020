#!/bin/bash
# Copyright 2021 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python split_data.py \
#     --tumor-path data/BC-TCGA-Tumor.txt \
#     --normal-path data/BC-TCGA-Normal.txt \
#     --seed 42 \
#     --test-proportion 0.2

python owkin-submission-training.py \
    --train-normal-alice data/BC-TCGA-Normal_client.csv \
    --train-tumor-alice data/BC-TCGA-Tumor_client.csv \
    --train-normal-bob data/BC-TCGA-Normal_server.csv \
    --train-tumor-bob data/BC-TCGA-Tumor_server.csv \
    --epsilon 1 5 10 --delta 1e-5 1e-4 \
    --output-dir owkin-models --subprocess

for file in owkin-models/*
do
    python owkin-submission-predict.py \
        --output-dir owkin-predictions \
        --model-path $file \
        --test-file data/test_samples.csv
done

for file in owkin-predictions/*
do
    echo $file
    python owkin-submission-evaluate.py \
        --labels data/test_labels.csv \
        --preds $file
done
