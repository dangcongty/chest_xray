import os
import sys

sys.path.append(os.getcwd())

from ultralytics import YOLO

# Load models
yolo11n_orig = YOLO('yolo11n.pt')
yolo11n_heat = YOLO('ultralytics/cfg/models/11/yolo11n-hm.yaml')

# Get state dictionaries
state_orig = yolo11n_orig.state_dict()
state_heat = yolo11n_heat.state_dict()

# Transfer compatible weights
for name, param in state_orig.items():
    if name in state_heat and state_heat[name].shape == param.shape:
        state_heat[name]=param
    else:
        print(name)
# Load the updated state dict into the new model
yolo11n_heat.load_state_dict(state_heat)

# (Optional) Save your initialized model
yolo11n_heat.save('ultralytics/cfg/models/11/yolo11n-hm-pretrained.pt')
print("✅ Transferred compatible weights and saved as 'yolo11n-hm-pretrained.pt'")
