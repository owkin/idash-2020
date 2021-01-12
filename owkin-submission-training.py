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
import sys
import docker
import subprocess
import pathlib

from itertools import product
from configargparse import ArgParser

TRAINING_IMAGE="owkin-submission:latest"
TRAINING_PROGRAM="src/convert_params_and_train.py"

def program_options():
    """Create argument parser for the CLI.
    """
    parser = ArgParser()
    dp_group = parser.add_argument_group(title="Required DP Parameters")
    io_group = parser.add_argument_group(title="Required Train Data Parameters")
    comm_group = parser.add_argument_group(title="Flags for Communication (no touch)")

    # File IO
    io_group.add(
        "--train-normal-alice",
        metavar="TRAIN-NORMAL-DATA-FILE",
        help="Path to training data file consisting of data samples corresponding "\
            "to the NORMAL classification label.",
        type=str,
        required=True
    )
    io_group.add(
        "--train-tumor-alice",
        metavar="TRAIN-TUMOR-DATA-FILE",
        help="Path to training data file consisting of data samples corresponding "\
            "to the TUMOR classification label.",
        type=str,
        required=True
    )
    io_group.add(
        "--train-normal-bob",
        metavar="TRAIN-NORMAL-DATA-FILE",
        help="Path to training data file consisting of data samples corresponding "\
            "to the NORMAL classification label.",
        type=str,
        required=True
    )
    io_group.add(
        "--train-tumor-bob",
        metavar="TRAIN-TUMOR-DATA-FILE",
        help="Path to training data file consisting of data samples corresponding "\
            "to the TUMOR classification label.",
        type=str,
        required=True
    )

    # DP Req. Options
    dp_group.add(
        "--epsilon", 
        nargs="+",
        help="Epsilon value(s) for differentially-private training. "\
            "One or many epsilon values can be specified. If multiple epsilons are "\
            "specified, then independent experiments will be run for each specified "\
            "epislon value. The results of each of these runs will be stored in "\
            "separate, named result files. "\
            "Epsilons can be specified as decimal values. Some examples of valid "\
            "epsilon arguments are "\
            "`--epsilon 3`, "\
            "`--epsilon 5.32341`, "\
            "`--epsilon 3 3.5 4 4.5 20`.",
        type=float,
        required=True
    )

    dp_group.add(
        "--delta",
        nargs="+",
        help= "Delta value(s) for differentially-private training. "\
            "One or many delta values can be specified. If multiple deltas are"\
            "specified, then independent experiments will be run for each delta"\
            "value in combination with each epsilon value."\
            "The reuslts of these runs are stored in separate, named result files."\
            "To use (eps)-DP for privacy calculations, pass use the option\n"\
            "`--delta 0`.",
        type=float,
        required=True
    )

    # Optional arguments
    parser.add(
        "--output-dir",
        help="Directory to store output trained model to. If the directory does not "\
            "exist, it will be created.",
        type=str,
        default="owkin-results"
    )

    # Next arguments
    comm_group.add(
        "--port",
        help="Specifies the port through which the two workers should communicate on the host machine.",
        default="8081",
        type=int
    )

    comm_group.add(
        "--subprocess",
        help="If set, the training will be performed between two subprocesses. If unset (default), "\
             "the training will be performed with Docker containers.",
        default=False,
        action="store_true"
    )
    args = parser.parse_args()

    # Some post processing, for lists etc.
    if not isinstance(args.epsilon, list):
        args.epsilon = [args.epsilon,]

    if not isinstance(args.delta, list):
        args.delta = [args.delta,]


    return args

def list_system_images(docker_client):
    """Return a list of all images on the current system.

    Args:
        docker_client (docker.client): An already initialized docker client

    Returns:
        list[str]: A list of strings of all image names/tags on the system.
    """
    return [
       image.attrs["RepoTags"][0]
       for image in docker_client.images.list()
       if len(image.attrs["RepoTags"]) > 0
    ]


def dict_to_cli_args(arg_dict):
    command_list = []
    for k, v in  arg_dict.items():
        if k in ["train_normal_alice", "train_tumor_alice", "train_normal_bob", "train_tumor_bob", "subprocess"]: 
            continue
        # 1. convert undercores to hypens
        arg = "--" + k.replace("_", "-")

        # 2. Add to list
        command_list.append(arg)
        if isinstance(v, list):
            for el in v:
                command_list.append(str(el))
        else:
            command_list.append(str(v))
    
    return command_list

def add_command(command_list, arg, value):
    """Inplace modification of the command list to append the given command.

    Args:
        command_list (list[str]): A list of flags/commands for the CLI
        arg (str): the name of the flag to append
        value (str): the value to give it
    """
    command_list.append(arg)
    # Make sure to add only strings. We are doing this in a maybe not 
    # robust way.
    if not isinstance(value, str):
        value = str(value)
    command_list.append(value)

def run_training_and_testing(args):
    """Run an entire training session for these training arguments.

    Args:
        training_args ([type]): [description]
    """
    # We are going to try just calling things on the base system without
    # docker for an instant.
    command_list = dict_to_cli_args(args)

    # Now make the separate command list for the two different runs
    alice_command_list = command_list.copy()
    bob_command_list = command_list.copy()

    # Finish Alice Commands
    add_command(alice_command_list, "--participant", "server")
    add_command(alice_command_list, "--training-seed", 141)
    add_command(alice_command_list, "--train-normal", args["train_normal_alice"])
    add_command(alice_command_list, "--train-tumor", args["train_tumor_alice"])

    # Finish Bob Commands
    add_command(bob_command_list, "--participant", "client")
    add_command(bob_command_list, "--training-seed", 42)
    add_command(bob_command_list, "--train-normal", args["train_normal_bob"])
    add_command(bob_command_list, "--train-tumor", args["train_tumor_bob"])

    if args["subprocess"]:
        print("Training with subprocesses")

        add_command(alice_command_list, "--mode", "subprocess")
        add_command(bob_command_list, "--mode", "subprocess")

        alice_run_command = ["python", TRAINING_PROGRAM] + alice_command_list
        bob_run_command = ["python", TRAINING_PROGRAM] + bob_command_list

        # Start Alice first
        print("* Starting Alice node...") 
        alice_proc = subprocess.Popen(alice_run_command)

        # Start Bob
        print("* Starting Bob node...")
        bob_proc = subprocess.call(bob_run_command)

    else:
        print("Training with docker")

        add_command(alice_command_list, "--mode", "docker")
        add_command(bob_command_list, "--mode", "docker")

        client = docker.from_env()

        # Check docker image existence
        if TRAINING_IMAGE not in list_system_images(client):
            print("The Docker image %s does not exist." % TRAINING_IMAGE)
            print("Please run the following command:")
            print("\t docker build . -t owkin-submission:latest")
            exit(1)

        container_output_dir = '/submission/%s' % args["output_dir"] 
        # Launch container for server
        client.containers.run(
            TRAINING_IMAGE,
            " ".join(alice_command_list),
            name="idash-server",
            detach=True,
            auto_remove=True,
            volumes={os.getcwd()+"/data": {'bind': '/submission/data', 'mode': 'ro'},
                     "%s/%s" % (os.getcwd(), args["output_dir"]): {'bind': container_output_dir, 'mode': 'rw'}}
        )
        server_ip = client.containers.get('idash-server').attrs['NetworkSettings']['IPAddress']
        # print("The IP of 'idash-server' is %s" % server_ip)

        # Launch container for client
        client.containers.run(
            TRAINING_IMAGE,
            "%s --host %s" % (" ".join(bob_command_list), server_ip),
            name="idash-client",
            auto_remove=True,
            volumes={os.getcwd()+"/data": {'bind': '/submission/data', 'mode': 'ro'},
                     "%s/%s" % (os.getcwd(), args["output_dir"]): {'bind': container_output_dir, 'mode': 'rw'}}
        )

        client.close()


if __name__ == "__main__":
    args = program_options()
    # print(args)

    # If we need an output directory, make sure it is there.
    os.makedirs(args.output_dir, exist_ok=True)

    run_training_and_testing(vars(args))

