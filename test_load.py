from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
root = os.path.expanduser("~/.cache/huggingface/lerobot/ozan/so100_fold_native")
ds = LeRobotDataset("ozan/so100_fold_native", root=root)
print("Loaded successfully, episodes:", len(ds.episodes) if ds.episodes else 0)
