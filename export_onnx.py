# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:01:32 2019

@author: rehakomoon
"""

from pathlib import Path
import torch
import torch.onnx

from model import Model

model_dir = Path("E:/vrc_rotation/log/")
export_dir = Path("E:/vrc_rotation/export/")
export_dir.mkdir(exist_ok=True)

model_epoch_list = [int(str(s)[-10:-4]) for s in model_dir.glob("model_*.pth")]
latest_model_path = model_dir / f"model_{max(model_epoch_list):06}.pth"
export_path = export_dir / (latest_model_path.stem + ".onnx")

print(f"load {latest_model_path}...")
model_params = torch.load(str(latest_model_path))

model = Model()
model.load_state_dict(model_params)
model.eval()

dummy_x = torch.randn(1, 3, 480, 480)
#dummy_y = model(dummy_x)

torch.onnx.export(model, dummy_x, str(export_path), verbose=True)