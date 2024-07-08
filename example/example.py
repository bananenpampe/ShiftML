import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))

from ShiftML import ShiftML, StandardOutput
import ase
from metatensor.torch.atomistic import(
    ModelOutput
)
import torch
from ase.build import bulk

frame = bulk("C", "diamond", a=3.566)
calc = ShiftML()
frame.set_calculator(calc)

output = frame.calc.run_model(frame, StandardOutput)
frame.arrays["cs_iso"] = output['mtt::cs_iso'].block(0).values.detach().numpy()