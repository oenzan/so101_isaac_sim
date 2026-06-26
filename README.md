# so101_isaac_sim

## Dataset Generation

To generate the SO-100 folding dataset using the native Isaac Sim physics pipeline, run the following command. The dataset will be generated and saved to huggingface locally under the repository ID `ozan/so100_fold_native`:

```bash
/home/ozan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh isaac_sim/scripts/generate_dataset_native.py
```

## Dataset Visualization

To visualize the generated dataset using `lerobot-dataset-viz`, use the following command. Make sure you are in the `lerobot` conda environment:

```bash
conda run -n lerobot lerobot-dataset-viz --repo-id ozan/so100_fold_native --episode-index 0
```
