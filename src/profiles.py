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

"""Creating and managing training profiles for different privacy levels
"""
import os
import pathlib
import pandas

# Known profiles, these are the run settings that give us the best predictive
# performance for a specified (eps, delta) privacy profile
PROFILE_FILE = pathlib.PurePosixPath(__file__).parent.joinpath("profiles.csv")
DROP_COLS = ["acc", "best_metric", "network"]


def nearest_leq_element(arr, target):
    """Return an item from the array which is the smallest value less than 
    or equal to the target.

    Args:
        arr ([type]): [description]
    """
    # Sublist of viable elementsa
    leq_elements = sorted([x for x in arr if x <= target])
    return leq_elements[-1]

def load_profiles(profile_file):
    """Load a profile file

    Args:
        profile_file (str): An absolute path to a profile file.
    """
    profile_df = pandas.read_csv(PROFILE_FILE)
    profile_df.drop(columns=DROP_COLS, inplace=True)

    # Sort all the privacy profiles such that last ones are
    # the least private (by index)
    profile_df.sort_values(
        ["delta", "epsilon"], 
        ascending=[True, True], 
        inplace=True, 
        ignore_index=True
    )

    return profile_df


def lookup_training_profile(epsilon, delta):
    """Return training parameters for a given (eps, delta) request.
    """
    # Read in profiles, they will be sorted according to eps/delta
    profile_df = load_profiles(PROFILE_FILE)

    # Now that we have the set of all profiles, we want to find the entry 
    # which is a best match in terms of privacy.

    # 1. Filter according to delta
    profile_df = profile_df[profile_df["delta"] <= delta]

    if len(profile_df) == 0:
        raise IndexError(f"There exists no appropraite satisfying profile for the requested. (delta={delta})")

    # 2. Filter according to epsilon
    profile_df = profile_df[profile_df["epsilon"] <= epsilon]

    if len(profile_df) == 0:
        raise IndexError(f"There exists no appropraite satisfying profile for the requested. (epsilon={epsilon}, delta={delta})")

    # Return the last element, which will be the privacy profile
    # closest to the one requested from below.
    final_profile = profile_df.iloc[-1].to_dict()

    # Remove non-argument keys
    del final_profile["epsilon"]
    del final_profile["delta"]

    return final_profile
    
    
     
