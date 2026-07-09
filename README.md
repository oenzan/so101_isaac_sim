# so101_isaac_sim

## Conda Environment Setup

To recreate the `foldnet` conda environment on another computer, first export the environment on the original machine:

```bash
conda env export > foldnet_environment.yml
```

Then copy `foldnet_environment.yml` to the new computer and create the environment with:

```bash
conda env create -f foldnet_environment.yml
```

Activate the environment:

```bash
conda activate foldnet
```

## Dataset Generation

To generate the SO-100 folding dataset using the native Isaac Sim physics pipeline, run the following command. The dataset will be generated and saved to Hugging Face locally under the repository ID `ozan/so100_fold_native`:

```bash
/home/ozan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh isaac_sim/scripts/generate_dataset_native.py
```

## Dataset Visualization

To visualize the generated dataset using `lerobot-dataset-viz`, use the following command. Make sure you are in the `lerobot` conda environment:

```bash
conda run -n lerobot lerobot-dataset-viz --repo-id ozan/so100_fold_native --episode-index 0
```
