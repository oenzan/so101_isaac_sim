import sys
import os
import torch
import numpy as np

foldnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code/external/batch_urdf/src"))
sys.path.insert(0, foldnet_src)
import batch_urdf
from batch_urdf.utils import quaternion_matrix

q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
print(quaternion_matrix(q))
