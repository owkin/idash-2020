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
import pathlib

# Generate relative paths to the proper genelists. 
GENELIST_ROTTERDAM=str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent,
            "..",
            "genes_selection",
            "genelist_rotterdam_GeneSymbolsForTCGA.txt"
        )
    ).resolve()
)

GENELIST_CITBCMST=str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent,
            "..",
            "genes_selection",
            "genelist_citbcmst_GeneSymbolsForTCGA.txt"
        )
    ).resolve()
)

def get_genes_list(file_path):
    with open(file_path, "r") as file_reader:
        genes = file_reader.readlines()
        genes = [gene.strip() for gene in genes]
        genes = [el for gene in genes for el in gene.split(" /// ")]
        genes = [el for gene in genes for el in gene.split("-")]
    return genes

def get_genome_signature(signature):
    if signature == "rotterdam":
        return get_genes_list(GENELIST_ROTTERDAM)
    elif signature == "citbcmst":
        return get_genes_list(GENELIST_CITBCMST)
    elif signature == "union":
        rotterdam_genes = get_genome_signature("rotterdam")
        citbcmst_genes = get_genome_signature("citbcmst")
        union_genes = list(set(rotterdam_genes) | set(citbcmst_genes))
        return union_genes
    else:
        print("Unknown genome signature %s." % signature)
        exit(1)

def genes_selection_extraction(X_train, signature):
    selected_genes = get_genome_signature(signature)
    genes = X_train.keys().tolist()
    selected_genes = list(set(selected_genes) & set(genes))
    selected_genes.sort() # Required for reproducibility
    X_train = X_train[selected_genes]
    return X_train

