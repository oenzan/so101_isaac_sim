import numpy as np

# A simple script to parse the URDF XML and print the cumulative Z/Y offsets
import xml.etree.ElementTree as ET

tree = ET.parse('/home/ozan/Downloads/so100_ws/isaac_sim/urdf/so_100_dual_foldnet.urdf')
root = tree.getroot()

for joint in root.findall('joint'):
    if joint.get('name') == 'left_tcp_joint':
        origin = joint.find('origin')
        if origin is not None:
            print("left_tcp_joint origin:", origin.get('xyz'))
