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

from pandas import concat, read_csv

GENES_COL_NAME="Hybridization REF"

def format_data(input_csv_path):
    data = read_csv(input_csv_path, sep="\t")
    header = data[GENES_COL_NAME]
    data.drop([GENES_COL_NAME], axis="columns", inplace=True)
    data = data.T
    data.columns = header
    return data

def create_test_dataset_without_split(data_file):
    """Create an ML-ready dataset from a gene data file.

    Note: different from the training dataset loader, this dat creation does not
    shuffle the data (key aspect in linking results to inputs).

    Args:
        data_file (str): Gene data file to run inference on.
        normalization (bool, optional): Normalizes samples if true. Defaults to False.
        fill_method (str, optional): Method for replacing missing data. Defaults to "mean".
    """
    # Read datsaet
    X = format_data(data_file)

    # Replace NaN
    X = X.fillna(0)

    return X

def create_dataset_without_split(tumor_csv_file, normal_csv_file, seed_shuffle=42):
    # Read and format datasets
    data_tumor = format_data(tumor_csv_file)
    data_normal = format_data(normal_csv_file)
    # Add predictions
    data_tumor["tumor"] = 1
    data_normal["tumor"] = 0
    data = concat([data_tumor, data_normal])
    # Shuffle dataset
    data = data.sample(frac=1, random_state=seed_shuffle)
    # Split into data and prediction
    y = data["tumor"].values
    X = data.drop(columns=["tumor"])
    X = X.fillna(0)
    return X, y

