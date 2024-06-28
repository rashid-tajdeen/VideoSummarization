# VideoSummarization

## Pre-requisite

Have the dataset from [here](https://videopipe.github.io/qvpipe/index.html) available in the dataset directory, and modify the `src/qvpipe_dataset.py` if needed.

## Usage Details

### Preprocess Data with Frame Extraction Methods
```shell
cd src
python3 frame_extraction.py
```

### Train and Test
```shell
cd src
python3 main.py --epochs 50 --classes 17 --num_frames 5 --train --valid --early_stopping --frame_selection random
```

#### Options
`--epochs`: Number of epochs for training.

`--classes`: 17 is the only implementation available.

`--num_frames`: Number of frames to represent a video.

`--train`: Flag to enable training.

`--valid`: Flag to enable testing.

`--early_stopping`: Flag to enable early stopping.

`--frame_selection`: Standalone options can be ["random", "less_blur", "motion", "k_means", "histogram", "entropy"]. Combination can also be used like ["less_blur motion", "motion k_means entropy", etc.,].