import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))

from model import ShiftML as ShiftMLModel
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput
)
import torch

#Output format for the Metatensor calculators
StandardOutput = {
    "mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True),
}

def ShiftML(filename="exported-model.pt"):
    '''
    Function to generate the Metatensor calculator suitable for ASE.
    input(optional):
        filename: the file to store the Metatensor calculator, default named "exported-model.pt"
    output:
        metatensor: the Metatensor calculator, which includes:
            model: `ShiftMLModel` defined in `model.py`
            metadata: information of this model
            capabilities: outputs information of the Metatensor calculator
    '''
    model = ShiftMLModel()
    weights = [torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", f"coeff{species_id}.pt")).to(torch.float32) for species_id in [1,6,7,8,16]]
    biases = [torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", f"bias{species_id}.pt")).to(torch.float32) for species_id in [1,6,7,8,16]]
    hypers_ps = {
        "cutoff": 4.6,
        "max_radial": 6,
        "max_angular": 6,
        "atomic_gaussian_width": 0.18,
        "center_atom_weight": 0.0,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width":0.5},
        },
        "radial_scaling":{"Willatt2018": {"exponent": 4.7, "rate": 2.0, "scale": 2.6}}
    }
    model.initialize_from_weights(weights, biases, [1,6,7,8,16,], hypers_ps)

    metadata = ModelMetadata(
        name="isotropic-chemical-shielding",
        description="a chemical shielding calculation model of the SOAP model",
        authors=["Yuxuan Zhang <yux.zhang@epfl.ch>"],
        references={
            # you can add references that should be cited when using this model here,
            # check the documentation for more information
        },
    )

    outputs = StandardOutput

    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=[1,6,7,8,16,],
        interaction_range=0.0,
        length_unit="",
        supported_devices=["cpu"],
        dtype="float32",
    )

    wrapper = MetatensorAtomisticModel(model.eval(), metadata, capabilities)

    # export to the pt file
    wrapper.save(filename)

    return MetatensorCalculator(filename)