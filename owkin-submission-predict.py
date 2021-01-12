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
import torch
import pandas as pd
import numpy as np

from src.models.logistic_regression_model import LogisticRegression
from src.utils.format_data import create_test_dataset_without_split
from src.utils.genes_selection import genes_selection_extraction
from src.utils.pytorch_evaluation import predict

def main(args):
    sizemodel = int(args.model_path.split("sizemodel")[1].split(".")[0])
    model = LogisticRegression(sizemodel).cpu()
    model.load_state_dict(torch.load(args.model_path))

    X_test = create_test_dataset_without_split(args.test_file)
    if sizemodel == 69:
        X_test = genes_selection_extraction(X_test, "rotterdam")
    else:
        X_test = genes_selection_extraction(X_test, "citbcmst")

    prediction_results = pd.DataFrame()
    prediction_results["patient_id"] = X_test.index.tolist()

    y_pred = predict(model, X_test)
    prediction_results["pred"] = np.squeeze(y_pred)

    output_file = os.path.basename(args.model_path).split("-sizemodel")[0].replace("model", "results") + ".csv"
    prediction_results.to_csv(
        os.path.join(args.output_dir, output_file),
        index=False
    )
    

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add(
        "--output-dir",
        help="Directory to store output result files to. If the directory does not "\
            "exist, it will be created.",
        type=str,
        default="."
    )

    parser.add(
        "--model-path",
        help="Path to the trained model.",
        type=str,
        required=True
    )

    parser.add(
        "--test-file",
        metavar="TEST-DATA-FILE",
        help="Path to test a test data file consisting of samples of unknown "\
            "classification. After DP-FL training is completed, the resulting "\
            "global model will be used to infer the tumor status of these samples. "\
            "The resulting predictions will be stored within the corresponding results "\
            "files.",
        required=True
    )

    args = parser.parse_args()

    # If we need an output directory, make sure it is there.
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


