#!/usr/bin/env python
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

import os

import configargparse
from pandas import concat, DataFrame, read_csv, Series

GENES_COL_NAME="Hybridization REF"

def split_dataframe(data_path, seed, test_proportion):
    data = read_csv(data_path, sep="\t")
    header = data[GENES_COL_NAME].tolist()
    data.drop([GENES_COL_NAME], axis="columns", inplace=True)

    data_test = data.sample(frac=test_proportion, random_state=seed, axis="columns")
    data.drop(data_test.keys().tolist(), axis="columns", inplace=True)
    data_test.insert(0, GENES_COL_NAME, header)

    data_server = data.sample(frac=0.5, random_state=seed, axis="columns")
    data.drop(data_server.keys().tolist(), axis="columns", inplace=True)
    data_server.insert(0, GENES_COL_NAME, header)
    data.insert(0, GENES_COL_NAME, header)

    data_server.to_csv(data_path.replace(".txt", "_server.csv"), sep='\t', index=False)
    data.to_csv(data_path.replace(".txt", "_client.csv"), sep='\t', index=False)

    return data_test

def main(args):
    data_tumor_test = split_dataframe(args.tumor_path, args.seed, args.test_proportion) 
    data_normal_test = split_dataframe(args.normal_path, args.seed, args.test_proportion)
    nb_tumor_samples = data_tumor_test.shape[1] - 1
    nb_normal_samples = data_normal_test.shape[1] - 1
    # Save test samples
    data_tumor_test.drop([GENES_COL_NAME], axis="columns", inplace=True)
    test_samples = concat([data_normal_test, data_tumor_test], ignore_index=False, axis="columns")
    test_samples.to_csv(os.path.join(os.path.dirname(args.tumor_path), "test_samples.csv"), sep='\t', index=False)
    # Save test labels
    patients = test_samples.columns.tolist()[1:]
    test_labels = DataFrame([[0] * nb_normal_samples + [1] * nb_tumor_samples], columns=patients)
    test_labels.to_csv(os.path.join(os.path.dirname(args.tumor_path), "test_labels.csv"), sep='\t', index=False)

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--tumor-path", help="Path to tumor file", type=str, required=True)
    parser.add("--normal-path", help="Path to normal file", type=str, required=True)
    parser.add("--seed", help="Seed used.", type=int, default=42)
    parser.add("--test-proportion", help="Test set proportion.", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
