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

import socket
import struct
import time

import torch

def start_server(port, mode="subprocess"):
    s = socket.socket()
    if mode == "docker":
        ip_address = socket.gethostbyname(socket.gethostname())
    elif mode == "subprocess":
        ip_address = "localhost"
    else:
        print("Unknown mode %s" % mode)
        exit(1)

    try:
       s.bind((ip_address, port))
    except OSError:
       time.sleep(1)
       s.bind((ip_address, port))

    s.listen(1)
    c, addr = s.accept()
    print("Connection from: " + str(addr))
    return c

def start_client(host, port):
    while True:
        try:
            s = socket.socket()
            s.connect((host, port))
            break
        except socket.error:
            print("Connection Failed, Retrying..")
            time.sleep(1)
    return s

def stop_server(conn):
    conn.close()

def send(conn, message):
    data = struct.pack('<%df' % len(message),*message)
    conn.send(data)

def receive(conn, size_list):
    data = conn.recv(1024)
    data = list(struct.unpack('<%df' % size_list ,data))
    return data

def send_ack(conn):
    message = "ok"
    conn.send(message.encode('utf-8'))

def receive_ack(conn):
    message = conn.recv(8).decode('utf-8')
    if message != "ok":
        print("Unrecognized acknowledgement.")
        exit(1)

def send_model(conn, model):
    data = torch.cat([weights.flatten() for weights in model.parameters()])
    send(conn, data.tolist())

def receive_model(conn, model, action):
    nb_params = 0
    for w in model.parameters():
        nb_params += w.numel()
    data = receive(conn, nb_params)
    weights = []
    idx = 0
    for w in model.parameters():
        weights.append(torch.FloatTensor(data[idx: idx+ w.numel()]).view(w.shape))
        idx += w.numel()
    if action == "overwrite":
        for w_new, w_old in zip(weights, model.parameters()):
            w_old.data.copy_(w_new)
    elif action == "aggregate":
        w_agg = [
            0.5 * (w_received + w_local) for w_received, w_local in zip(weights, model.parameters())
        ]
        for w_new, w_old in zip(w_agg, model.parameters()):
            w_old.data.copy_(w_new)
    else:
        print("Unknown action %s" % action)
        exit(1)
