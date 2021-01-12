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

import random

from warnings import filterwarnings
import configargparse

import numpy as np
import pandas as pd
import torch
from opacus.privacy_engine import PrivacyEngine

from utils.format_data import create_dataset_without_split
from utils.genes_selection import genes_selection_extraction
from models.logistic_regression_model import LogisticRegression
from utils.communication import start_server, stop_server, start_client, send_model, receive_model, send_ack, receive_ack

filterwarnings('ignore')

def training(args, conn):
    # Create datasets 
    X_train, y_train = create_dataset_without_split(
        args["train_tumor"], 
        args["train_normal"], 
        42
    )

    # Feature Extraction
    if args["genes_selection"] != "None":
        X_train = genes_selection_extraction(X_train, args["genes_selection"])

    # Initialize all seeds for reproducibility
    seed = args['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = LogisticRegression(X_train.shape[1])
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=args['sample_rate'],
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args['noise_multiplier'],
        max_grad_norm=args['max_grad_norm'],
        secure_rng=True
    )
    privacy_engine.attach(optimizer)

    samples = X_train.to_numpy(dtype="float32")
    labels = y_train.astype("float32")

    if args['fl_strategy'] == "walk":
        model, optimizer = walk_training(samples, labels, model, optimizer, criterion, args['participant'], conn,
                                     args['fl_rounds'], args['batches_per_round'], args['sample_rate'])
    else:
        print("Unkown strategy %s." % args['fl_strategy'])
        exit(1)

    if args["participant"] == "server":
        model = model.cpu()
        torch.save(model.state_dict(),  "server_model.pth")

    return X_train.shape[1]


def walk_training(samples, labels, model, optimizer, criterion, participant, conn,
                  fl_rounds, batches_per_round, sample_rate):
    model.train()
    if participant == "client":
        send_model(conn, model)

    for fl_round in range(fl_rounds):
        for _ in range(batches_per_round):
            receive_model(conn, model, "overwrite")
            # Create batch
            mask = np.random.uniform(0, 1, len(samples)) < sample_rate
            x = torch.from_numpy(samples[mask])
            y = torch.from_numpy(labels[mask])
            # Forward pass
            y_pred = model(x)
            # Compute Loss
            loss = criterion(y_pred, y)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            send_model(conn, model)

    if participant == "server":
        receive_model(conn, model, "overwrite")
    return model, optimizer

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--genes-selection", help="Selection of genes.",
               choices=["rotterdam", "citbcmst"], default="rotterdam")
    parser.add("--learning-rate", help="Learning rate.", type=float, default=0.01)
    parser.add("--sample-rate", help="Proba to select each sample in a batch.",
               type=float, default=0.5)
    parser.add("--training-seed", help="Seed used for training.", type=int, default=42)

    parser.add("--noise-multiplier", help="(DP) Noise multiplier.", type=float, default=1.3)
    parser.add("--max-grad-norm", help="(DP) Clipping threshold.", type=float, default=5.0)
    parser.add("--delta", help="(DP) Target delta.", type=float, default=1e-5)

    parser.add("--fl-strategy", help="(FL) FL strategy.",
               choices=["walk"], default="walk")
    parser.add("--fl-rounds", help="(FL) Number of FL rounds (aggregations).", type=int, default=5)
    parser.add("--batches-per-round", help="(FL) Number of batch updates in one FL round.", type=int, default=1)

    parser.add("--participant", choices=["client", "server"], required=True)
    parser.add("--train-tumor", help="Path to train tumor file", type=str, required=True)
    parser.add("--train-normal", help="Path to train normal file", type=str, required=True)

    parser.add("--host", help="Server IP address", type=str, default="localhost")
    parser.add("--port", help="Server port", type=int, default=8080)
    parser.add("--mode", help="Launching mode", type=str, 
               choices=["subprocess", "docker"], default="subprocess")

    args = parser.parse_args()

    # Startup networking
    if args.participant == "server":
        conn = start_server(args.port, mode=args.mode)
    else:
        conn = start_client(args.host, args.port)

    training(vars(args), conn)

    if args.participant == "server":
        stop_server(conn)
