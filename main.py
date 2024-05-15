"""
This script is not done yet
"""

import os
import torch
from src.models.transformer import Network


model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'chess-vision/src/trained_models/model.pkl')

checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

model = Network()
model.load_state_dict(checkpoint['model'])

xb = torch.tensor([[1, 2, 3,4, 5,6, 7,8, 9, 10, 10, 10,10]])

model(xb)