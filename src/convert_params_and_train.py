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
import pathlib
import time
import shutil

from itertools import product
from configargparse import ArgParser

import profiles
import distant
from utils.communication import start_server, stop_server, start_client

DISTANT_OUTPUT_FILE="server_model.pth"

def program_options():
    """Create argument parser for the CLI.
    """
    parser = ArgParser()
    dp_group = parser.add_argument_group(title="Required DP Parameters")
    io_group = parser.add_argument_group(title="Required Train Data Parameters")

    # File IO
    io_group.add(
        "--train-normal",
        metavar="TRAIN-NORMAL-DATA-FILE",
        help="Path to training data file consisting of data samples corresponding "\
            "to the NORMAL classification label.",
        type=str,
        required=True
    )
    io_group.add(
        "--train-tumor",
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

    parser.add(
        "--participant",
        metavar="PARTICPANT",
        help="Server or client.",
        choices=["server", "client"],
        required=True
    )

    parser.add(
        "--training-seed",
        help="Seed used for the training.",
        type=int,
        default=42
    )

    parser.add(
        "--output-dir",
        help="Directory to store output result files to. If the directory does not "\
            "exist, it will be created.",
        type=str,
        default="."
    )

    parser.add(
        "--host",
        help="Specifies the server address.",
        default="localhost",
        type=str
    )

    parser.add(
        "--port",
        help="Specifies the port through which the two workers should communicate on the host machine.",
        default="8081",
        type=int
    )

    parser.add(
        "--mode",
        help="Specifies the launching mode.",
        default="subprocess",
        choices=["subprocess", "docker"],
        type=str
    )

    args = parser.parse_args()

    # Some post processing, for lists etc.
    if not isinstance(args.epsilon, list):
        args.epsilon = [args.epsilon,]

    if not isinstance(args.delta, list):
        args.delta = [args.delta,]


    # Convert all relative file paths into absolute paths
    resolve_path = lambda p: str(pathlib.Path(p).resolve())
    args.train_normal = resolve_path(args.train_normal)
    args.train_tumor = resolve_path(args.train_tumor)
    if args.output_dir is not None:
        args.output_dir = resolve_path(args.output_dir)

    return args

def run_training_and_testing(delta, prog_args, training_args, conn):
    """Runt an entire training session for these training arguments.

    Args:
        training_args ([type]): [description]
    """

    concat_args = {}
    for key in prog_args:
        concat_args[key] = prog_args[key]
    for key in training_args:
        concat_args[key] = training_args[key]
    concat_args["delta"] = delta
    del concat_args["epsilon"]
    return distant.training(concat_args, conn)


if __name__ == "__main__":
    args = program_options()

    if args.participant == "server":
        # If we need an output directory, make sure it is there.
        os.makedirs(args.output_dir, exist_ok=True)

    # Startup networking
    if args.participant == "server":
        conn = start_server(args.port, mode=args.mode)
    else:
        conn = start_client(args.host, args.port)

    for epsilon, delta in product(args.epsilon, args.delta):
        print(f"Training for profile (eps={epsilon}, delta={delta})")
        train_args = profiles.lookup_training_profile(epsilon, delta)

        # Now we just need to punch in this training call.
        # launch distant script with the correct parameters
        sizemodel = run_training_and_testing(delta, vars(args), train_args, conn)

        # From here, we now need to move the output to the right result
        # directories.
        if args.participant == "server":
            shutil.move(
                DISTANT_OUTPUT_FILE,
                os.path.join(
                    args.output_dir,
                    f"owkin-model-eps{epsilon}-delta{delta}-sizemodel{sizemodel}.pth"
                )
            )

    if args.participant == "server":
        stop_server(conn)
