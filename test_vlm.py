import sys, os
from PIL import Image

sys.path.insert(0, "/home/vision/work/aura")
from analyzer.inference import MiniInternVL2DriveLMDescriber
import torch

describer = MiniInternVL2DriveLMDescriber()
print("Model loaded.")

# create dummy image
img = Image.new('RGB', (640, 480), color = 'red')

print("Running describe...")
try:
    res = describer.describe(img, prompt="corridor nav")
    print("Result:", res)
except Exception as e:
    print("Exception:", e)
