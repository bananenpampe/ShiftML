import torch
from typing import Dict, List, Optional
from metatensor.torch import Labels, TensorMap
from rascaline.torch.calculators import SoapPowerSpectrum
from metatensor.torch.learn.nn import Linear
from itertools import combinations_with_replacement
from metatensor.torch.atomistic import (
    ModelOutput,
    System,
)

class ShiftML(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = None
        self.unique_species = None
        self.hypers = None
        self.calculator = None

    def initialize_from_weights(self, 
                                weight_list: List[torch.Tensor],
                                bias_list: List[torch.Tensor],
                                unique_species: List[int],
                                hypers: dict,
                                ):
        self.unique_species = torch.unique(torch.tensor(unique_species, dtype=torch.int32))
        self.hypers = hypers
        
        soap_size = ((len(unique_species) * (len(unique_species) + 1) // 2)
            * hypers["max_radial"] ** 2
            * (hypers["max_angular"] + 1))

        l_tmp = Labels("cs_iso", torch.tensor([[0],]))
        soap_labels = Labels("center_type", self.unique_species.reshape(-1,1))

        self.linear = Linear(soap_labels, soap_size, 1, [l_tmp for _ in unique_species], dtype=torch.float32)
        self.calculator = SoapPowerSpectrum(**self.hypers)
        
        state_dict = self.linear.state_dict()

        # loop over weights and modify state dict
        for n, species_id in enumerate(self.unique_species):
            coeffs = weight_list[n]
            bias = bias_list[n]
            state_dict[f"module_map.{n}.weight"] = torch.reshape(coeffs, (1, soap_size))
            state_dict[f"module_map.{n}.bias"] = torch.reshape(bias, (1,))

        # set modified state dict
        self.linear.load_state_dict(state_dict)

        filter_by_central_id = 6

        neighbours_idx = list(combinations_with_replacement(self.unique_species.tolist() ,r=2))
        neighbours_idx = torch.tensor(neighbours_idx)
        self.unique_pairs = Labels(["neighbor_1_type", "neighbor_2_type" ], neighbours_idx)
        
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # compute the energy for each system by adding together the energy for each atom
        
        # get expanded soap
        Xsoap = self.calculator(systems).keys_to_properties(self.unique_pairs)

        out = self.linear(Xsoap).keys_to_samples("center_type")

        # write a function
        return {
            "mtt::cs_iso": out
        }
    