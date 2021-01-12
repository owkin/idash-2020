# iDASH 2020 Track 3 Submission -- Owkin

This repository contains Owkin's Track 3 (federated differential privacy)
submission for the iDash 2020 competition.
The details of the submission are presented in [our article](https://arxiv.org/pdf/2101.02997.pdf).

## Installation and Configuration

There are two types of installations for the two different modes of this submission: 
subprocess and docker.
Subprocess means the experiments runs on two subprocesses on the host machine,
docker means the experiment simulates the real setting more closely by spawning
multiple containers and handling the communication.

Both are described below:

### Base System Setup and Configuration

To run the submission on your base system, we suggest creating a clean 
environment using either Conda or PyEnv (with the virtualenv plugin).

**Conda**

```bash
$ conda create -n owkin-submission python=3.7.7
...
$ conda activate owkin-submission
```

**PyEnv**

```bash
$ pyenv virtualenv 3.7.7 owkin-submission
...
$ pyenv activate owkin-submission
```

With the new environment activated, the submission dependencies may be installed
via `pip`.

```bash
$ pip install -r requirements.txt
```
You should now be able to run the submission program. 

### Docker Setup and Configuration

To use the containerized version of the submission, the Desktop version of
Docker must be installed on the base system. Docker is available for multiple
operating systems; Installation instructions for your system 
[can be found here](https://www.docker.com/get-started).

With Docker installed, one simply needs to build the submission image:

```bash
$ docker build . -t owkin-submission:latest
```

You will now be able to run the submission within a Docker container (see
next section for details).

## Training Submission Program Description

With the setup and configuration out of the way, you should now be able to run the
training submission program. The program accepts as inputs:

- One (or multiple) epsilon specifications,
- One (or multiple) Delta specifications [0, or exponent],
- A per worker training dataset files (CSV, including labels),
- A output directory

The program will output:

- One file (.pth) containing the model per specified (epsilon, delta).
The epsilon and the delta values used for the training will be written in the filename
(e.g. owkin-model-eps1.0-delta0.0001-sizemodel69.pth).

### Usage

```
$ python owkin-submission-training.py --help
usage: owkin-submission-training.py [-h] --train-normal-alice
                                    TRAIN-NORMAL-DATA-FILE --train-tumor-alice
                                    TRAIN-TUMOR-DATA-FILE --train-normal-bob
                                    TRAIN-NORMAL-DATA-FILE --train-tumor-bob
                                    TRAIN-TUMOR-DATA-FILE --epsilon EPSILON
                                    [EPSILON ...] --delta DELTA [DELTA ...]
                                    [--output-dir OUTPUT_DIR] [--port PORT]
                                    [--subprocess]

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory to store output trained model to. If the
                        directory does not exist, it will be created.

Required DP Parameters:
  --epsilon EPSILON [EPSILON ...]
                        Epsilon value(s) for differentially-private training.
                        One or many epsilon values can be specified. If
                        multiple epsilons are specified, then independent
                        experiments will be run for each specified epislon
                        value. The results of each of these runs will be
                        stored in separate, named result files. Epsilons can
                        be specified as decimal values. Some examples of valid
                        epsilon arguments are `--epsilon 3`, `--epsilon
                        5.32341`, `--epsilon 3 3.5 4 4.5 20`.
  --delta DELTA [DELTA ...]
                        Delta value(s) for differentially-private training.
                        One or many delta values can be specified. If multiple
                        deltas arespecified, then independent experiments will
                        be run for each deltavalue in combination with each
                        epsilon value.The reuslts of these runs are stored in
                        separate, named result files.To use (eps)-DP for
                        privacy calculations, pass use the option `--delta 0`.

Required Train Data Parameters:
  --train-normal-alice TRAIN-NORMAL-DATA-FILE
                        Path to training data file consisting of data samples
                        corresponding to the NORMAL classification label.
  --train-tumor-alice TRAIN-TUMOR-DATA-FILE
                        Path to training data file consisting of data samples
                        corresponding to the TUMOR classification label.
  --train-normal-bob TRAIN-NORMAL-DATA-FILE
                        Path to training data file consisting of data samples
                        corresponding to the NORMAL classification label.
  --train-tumor-bob TRAIN-TUMOR-DATA-FILE
                        Path to training data file consisting of data samples
                        corresponding to the TUMOR classification label.

Flags for Communication (no touch):
  --port PORT           Specifies the port through which the two workers
                        should communicate on the host machine.
  --subprocess          If set, the training will be performed between two
                        subprocesses. If unset (default), the training will be
                        performed with Docker containers.
```

### Example Run

The basic configuration of the submission program is to run the submission for
a single value of $\epsilon$. This can be accomplished either running the 
script on the base system, or by running it from a container. 

**$(\epsilon, \delta)-DP Run**
```
$ python owkin-submission-training.py \
    --train-normal-alice data/BC-TCGA-Normal_client.csv \
    --train-tumor-alice data/BC-TCGA-Tumor_client.csv \
    --train-normal-bob data/BC-TCGA-Normal_server.csv \
    --train-tumor-bob data/BC-TCGA-Tumor_server.csv \
    --epsilon 1 --delta 1e-5 \
    --output-dir owkin-models
```

### Multi-Epsilon Run

```bash
$ python owkin-submission-training.py ... --epsilon 3 5 10 15 20 25 30 ...
```

### Run with Docker containers or subprocesses

By default, it runs with Docker containers.
To run with subprocesses, we have to add the 'subprocess' argument.

```bash
$ python owkin-submission-training.py ... --subprocess
```

## Predict Submission Program Description

With the setup and configuration out of the way, you should now be able to run the
predict submission program. The program accepts as inputs:

- One trained model file path,
- A single test dataset file (CSV, no labels included).
- A output directory

The program will output:

- One result file (CSV).

### Usage

```
$ python owkin-submission-predict.py  --help
usage: owkin-submission-predict.py [-h] [--output-dir OUTPUT_DIR] --model-path
                                   MODEL_PATH --test-file TEST-DATA-FILE

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory to store output result files to. If the
                        directory does not exist, it will be created.
  --model-path MODEL_PATH
                        Path to the trained model.
  --test-file TEST-DATA-FILE
                        Path to test a test data file consisting of samples of
                        unknown classification. After DP-FL training is
                        completed, the resulting global model will be used to
                        infer the tumor status of these samples. The resulting
                        predictions will be stored within the corresponding
                        results files.
```

### Example Run

```
$ python owkin-submission-predict.py \
    --output-dir owkin-predictions \
    --model-path owkin-models/owkin-model-eps1.0-delta0.0001-sizemodel69.pth \
    --test-file data/test_samples.csv
```

### Result File Format

The output result files contain binary values corresponding to predictions of a
trained model on each test data sample, where 

- `0` -- indicates a non-tumor prediction,
- `1` -- indicates a tumor prediction.

Each result CSV file will contain as many rows as test samples, and will contain
one (or multiple) columns, where each column represents a different independent
trial of the training procedure. Below you can find an example of the output
format

```csv
$ cat owkin-results-eps1.0-delta0.0001.csv
patient_id,pred
TCGA-BH-A0AY-11A-23R-A089-07, 0
TCGA-BH-A0DK-11A-13R-A089-07, 1
TCGA-A7-A13F-11A-42R-A12P-07, 0
...
TCGA-BH-A1ES-01A-11R-A137-07, 1
```

If you have a separate label file available, we provide a helper script to 
compare the accuracy of our outputs to your label file,

```bash
$ python owkin-submission-evaluate.py \
    --labels data/test_labels.csv \
    -- preds owkin-results-eps1.0-delta0.0001.csv
Accuracy: 0.9585
```

## License

This project is developed under the Apache License, Version 2.0 (Apache-2.0), located in the [LICENSE](./LICENSE) file.
