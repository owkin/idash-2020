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

import torch
import numpy as np


def predict(model, X, threshold=0.5):
    """Generate NumPy output predictions on a dataset using a given model.

    Args:
        model (torch model): A Pytroch model
        X (dataloader): A dataframe-based gene dataset to predict on
    """
    X_tensor, _  = convert_dataframe_to_tensor(X, [])

    model.eval()
    with torch.no_grad():
        y_pred = (model(X_tensor) >= threshold).int().numpy()

    return y_pred

def convert_dataframe_to_tensor(X, y):
    tensor_x = torch.Tensor(X.to_numpy().astype(np.float32))
    tensor_y = torch.Tensor(y)
    return tensor_x, tensor_y

