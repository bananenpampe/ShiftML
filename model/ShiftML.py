import sys
import os
import ase.io
from rascaline.torch.calculators import SoapPowerSpectrum
import numpy as np
import random
import pickle
SEED = 0
random.seed(SEED)
from sklearn.metrics import mean_squared_error
from rascaline.torch import systems_to_torch
import torch
from itertools import combinations_with_replacement
import metatensor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

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

def get_unique_species(frames):
    '''
    Function that gets the species list of a given list of structures, returns an array.
    '''
    if len(frames) == 1:
        frames = [frames]
    u_i = []
    for frame in frames:
        u_i.extend(np.unique(frame.get_atomic_numbers()).tolist())
    return np.unique(u_i)

def get_chemical_shielding(frames, 
           species=6, 
           identifier="cs_iso"):
    """Function that extracts chemical shieldings from list of ase.Atoms objetcs"""
    cs = []
    for frame in frames:
        cs.append(frame.arrays[identifier][frame.get_atomic_numbers() == species])
    return np.hstack(cs)

def densify(feat):
    '''
    Function that converts the results from the calculator to the arrays for fitting.
    '''
    return feat.keys_to_properties(["neighbor_1_type", "neighbor_2_type" ])

def convert_from_ase(calculator, frames, species_id=6):
    '''
    Function that converts the structure lists to arrays for fitting.
    '''
    unique_species_train = get_unique_species(frames)
    #First list out elements from ase
    frames = systems_to_torch(frames)
    
    #List out the table for the neighbor list
    neighbours_idx = torch.tensor(list(combinations_with_replacement(unique_species_train, r=2)))

    #Convert to the data from the SOAP power spectrum
    central_idx = torch.ones_like(neighbours_idx[:,0]).reshape(-1,1) * species_id
    labels_vals = torch.cat([central_idx,neighbours_idx],dim=1)
    filtered_by = metatensor.torch.Labels(["center_type", "neighbor_1_type", "neighbor_2_type" ], labels_vals)
    X = densify(calculator(frames, selected_keys=filtered_by)).block(0).values.numpy()
    return X

def get_frame_groups(frames, filter_by_central_id=6):
    ids = []
    for n, frame in enumerate(frames):
        ids_tmp = np.ones(len(frame))*n
        if filter_by_central_id is not None:
            ids_tmp = ids_tmp[frame.get_atomic_numbers() == filter_by_central_id]
        ids.append(ids_tmp)
    return np.hstack(ids)

def cross_validation(frames, X, Y, alpha=np.logspace(-6, 3, 30),species=6):
    G_train = get_frame_groups(frames, species)
    splits = list(GroupKFold(n_splits=5).split(X, Y, groups=G_train))
    model = RidgeCV(alphas=np.logspace(-6,3,30),cv=splits,scoring="neg_mean_squared_error")
    model.fit(X, Y)
    return model

def fit_parameters(hypers_ps, 
                   train_file, 
                   test_file, 
                   output_directory,
                   random_shuffle_train=True, 
                   random_shuffle_test=False, 
                   trunk_train=False, trunk_test=False,
                   ntrain = 1000, ntest=100, 
                   filter_train=True, filter_test=False):
    '''
    Functions that reads the hyperparameters and inputs xyz data
    inputs:
        `hyper_ps`: dict type, the hyperparameters of the SOAP model
        `train_file`: file name of the train datasets
        `test_file`: file name of the test datasets
        `random_shuffle_train`: option whether the train datasets are shuffled, default True
        `random_shuffle_test`: option whether the test datasets are shuffled, default False
        `trunk_train`: option whether the train dataset should be trunked, default False
        `trunk_test`: option wheter the test dataset should be trunked, default True
        `ntrain`: number of train datasets to take if trunk_train is True, default 1000
        `ntest`: number of test datasets to take if trunk_test is True, default 100
        `filter_train`: whether filtering the train datasets is necessary, default True,
    pass the argument with structure.info["STATUS"]=="PASSING"
        `filter_test`: whether filtering the test datasets is necessary, default False,
    pass the argument with structure.info["STATUS"]=="PASSING"
    
    outputs:
        save torch files of the linear coefficients and intercepts in `.pt` format
        record 


    '''

    #Load files
    frames_train_raw = ase.io.read(train_file,":")
    frames_test_raw = ase.io.read(test_file,":")
    if random_shuffle_train:
        random.shuffle(frames_train_raw)
    if random_shuffle_test:
        random.shuffle(frames_test_raw)
    if filter_train:
        frames_train = []
        for frame in frames_train_raw:
            if frame.info["STATUS"] == "PASSING": 
                frames_train.append(frame)
    else:
        frames_train = frames_train_raw
    if filter_test:
        frames_test = []
        for frame in frames_test_raw:
            if frame.info["STATUS"] == "PASSING": 
                frames_test.append(frame)
    else:
        frames_test = frames_test_raw
    if trunk_train:
        frames_train = frames_train[:ntrain]
    if trunk_test:
        frames_test = frames_test[:ntest]

    #Calculations
    calculator = SoapPowerSpectrum(**hypers_ps)
    species = get_unique_species(frames_train)
    record_error = open(os.path.join(output_directory, "error_estimation.txt", "w"))
    for species_id in species:
        X_train = convert_from_ase(calculator, frames_train, species_id=species_id)
        Y_train = get_chemical_shielding(frames_train, species=species_id)
        model = cross_validation(frames_train, X_train, Y_train, species=species_id)
        X_test = convert_from_ase(calculator, frames_test, species_id=species_id)
        Y_test = get_chemical_shielding(frames_test, species=species_id)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred, squared=False)
        print(f"rmse of species {species_id}:", mse, file=record_error)
        print(f"rmse of species {species_id}:", mse)
        print(f"calculating species {species_id} completed")
        bias = torch.tensor(model.intercept_, dtype=torch.float64)
        coeffs = torch.tensor(model.coef_, dtype=torch.float64)
        pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", f"bias{species_id}.pt")
        torch.save(bias, pathname)
        pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", f"coeff{species_id}.pt")
        torch.save(coeffs, pathname)
    record_error.close()
