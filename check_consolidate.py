from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
root = os.path.expanduser("~/.cache/huggingface/lerobot/ozan/so100_fold_native")
ds = LeRobotDataset.create("ozan/so100_fold_native", root=root, fps=30, features={"action": {"dtype": "float32", "shape": (6,)}})
ds.add_frame({"action": [0,0,0,0,0,0]})
ds.save_episode()
ds.consolidate()
print("Consolidated!")
