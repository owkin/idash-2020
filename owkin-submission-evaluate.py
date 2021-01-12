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

from statistics import mean
from numpy import std
import configargparse
from pandas import read_csv
from pandas import concat
from sklearn.metrics import accuracy_score

def main(args):
    labels = read_csv(args.labels, sep="\t")
    labels = labels.T
    preds = read_csv(args.preds, sep=",")
    preds.set_index("patient_id", inplace=True)
    trials = preds.columns.tolist()
    # Concat dataframes to be sure that the patients are in the same order
    df = concat([labels, preds], axis="columns")

    scores = []
    y_true = df[0].tolist()
    for trial in trials:
        y_pred = df[trial].tolist()
        scores.append(accuracy_score(y_true, y_pred))

    if len(scores) > 1:
        print("Accuracy: %0.4f (+/- %0.4f)" %  (mean(scores), std(scores)))
    else:
        print("Accuracy: %0.4f" %  scores[0])

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--labels", help="Path to labels file", type=str, required=True)
    parser.add("--preds", help="Path to predictions file", type=str, required=True)
    args = parser.parse_args()

    main(args)
