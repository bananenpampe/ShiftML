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
import os


def get_unique_species(frames):
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
    return feat.keys_to_properties(["neighbor_1_type", "neighbor_2_type" ])

def convert_from_ase(calculator, frames, species_id=6):
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

if __name__ == "__main__":
    pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "CSD-3k+S546_shift_tensors.xyz")
    frames_train_raw = ase.io.read(pathname,":")
    pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "CSD-500+104-7_shift_tensors.xyz")
    frames_test = ase.io.read(pathname,":")
    random.shuffle(frames_train_raw)
    #frames_train_raw = frames_train_raw[:NTRAIN]
    frames_train = []
    for frame in frames_train_raw:
        if frame.info["STATUS"] == "PASSING": 
            frames_train.append(frame)

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
    calculator = SoapPowerSpectrum(**hypers_ps)
    species = get_unique_species(frames_train)
    pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", "error_estimation.txt")
    record_error = open(pathname, "w")

    for species_id in species:
        X_train = convert_from_ase(calculator, frames_train, species_id=species_id)
        #G_train = get_frame_groups(frames_train, filter_by_central_id=species_id)
        Y_train = get_chemical_shielding(frames_train, species=species_id)
        model = cross_validation(frames_train, X_train, Y_train, species=species_id)
        X_test = convert_from_ase(calculator, frames_test, species_id=species_id)
        Y_test = get_chemical_shielding(frames_test, species=species_id)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred, squared=False)
        print(f"rmse of species {species_id}:", mse, file=record_error)
        pathname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "parameter_files", f"element{species_id}.pkl")
        with open(pathname, "wb") as f:
            pickle.dump(model, f)
        print(f"calculating species {species_id} completed")
        
    record_error.close()

