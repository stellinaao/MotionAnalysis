# imports
import pandas as pd
import numpy as np
from scipy.stats import zscore

# def global vars
bodyparts = np.array(["ear_tip", "ear_base", "ear_bottom", "pupil", "nose_tip", "nose_corner", "tongue", "fr_paw", "fl_paw"])
subject_ids = np.array(["MR15", "MR17", "MR19", "MR20", "MR26", "MR35", "MR47", "MR52"])
session_ids = np.array(["20221006_125935", "20221006_160204", "20221212_113618", "20221212_152230", "20230425_130517", "20230808_124302", "20240408_125103", "20240408_145925"])
n_subj = len(subject_ids)
n_bodyparts = len(bodyparts)
model_name = "model1-lite"
fs = 30

def read_data():
    dlc_outputs = []
    for subj_id, sess_id in zip(subject_ids, session_ids):
        dlc_outputs.append(read_sess(subj_id, sess_id, bodyparts, model_name))
    return dlc_outputs

# def functions
def read_sess(subject_id, session_id, bodyparts, model_name):
    df = pd.read_csv(f"data/{model_name}/{subject_id}_DynamicForaging_{session_id}_cam0_00000000DLC_resnet50_MR15Jul8shuffle1_300000.csv", skiprows=[0, 2])

    for part in bodyparts:
        df.rename(columns={f"{part}": f"{part}_x", f"{part}.1": f"{part}_y", f"{part}.2": f"{part}_p"}, inplace=True)

    return df

def remove_bp(dlc_outputs, bp2rem):
    mask = [part not in bodyparts for part in bp2rem]
    if any(mask):
        raise ValueError(f"ERROR: {bp2rem[mask]} not in list of bodyparts.")

    cols_2rem = np.ravel([[f"{part}_x", f"{part}_y", f"{part}_p"] for part in bp2rem])
    set_bodyparts(np.setdiff1d(bodyparts, bp2rem))
    set_n_bodyparts(len(bodyparts))
    return [dlc_output.drop(columns=cols_2rem) for dlc_output in dlc_outputs]

def set_bodyparts(new_val):
    global bodyparts 
    bodyparts = new_val
    
def set_n_bodyparts(new_val):
    global n_bodyparts
    n_bodyparts = new_val

def get_coords(dlc_outputs):
    bodypart_coords = [f"{part}_{coord}" for part in bodyparts for coord in ["x", "y"]]
    dlc_output_coords = [dlc_output.filter(items=bodypart_coords) for dlc_output in dlc_outputs]

    return dlc_output_coords

def mean_imputation(dlc_outputs, thresh, verbose=False):
    dlc_outputs_impute = [df.copy() for df in dlc_outputs]
    assert all([(dlc_outputs_impute[i].values == dlc_outputs[i].values).all() for i in range(len(dlc_outputs))])

    for i, dlc_output in enumerate(dlc_outputs_impute):
        for part in bodyparts:
            x = f"{part}_x"; y = f"{part}_y"; p = f"{part}_p"
            coords = dlc_output[[x, y]]
            prev_mean_x = np.round(dlc_output[x].mean(), 6)
            prev_mean_y = np.round(dlc_output[y].mean(), 6)

            mask = dlc_output[p] >= thresh
            valid = coords.loc[dlc_output.index[mask].tolist()]

            mean_x = valid[x].mean()
            mean_y = valid[y].mean()

            coords.loc[dlc_output.index[~mask].tolist()] = [mean_x, mean_y]
            dlc_output[[x, y]] = coords

            curr_mean_x = np.round(dlc_output[x].mean(), 6)
            curr_mean_y = np.round(dlc_output[y].mean(), 6)

            if verbose and not(curr_mean_x == np.round(mean_x, 6) and curr_mean_y == np.round(mean_y, 6)):
                print(f"{curr_mean_x}, {curr_mean_y}, {np.round(mean_x, 6)}, {np.round(mean_y, 6)}, {prev_mean_x}, {prev_mean_y}")

            assert (curr_mean_x == np.round(mean_x, 6) and curr_mean_y == np.round(mean_y, 6))

    return dlc_outputs_impute

def norm_session_bp(dlc_outputs, verbose=False):
    for i, dlc_output in enumerate(dlc_outputs):
        for j, part in enumerate(bodyparts):
            x = f"{part}_x"; y = f"{part}_y"
            dlc_output[x] = zscore(dlc_output[x])
            dlc_output[y] = zscore(dlc_output[y])

            if verbose: print(f"{np.round(dlc_output[x].mean(),6)}, {np.round(dlc_output[y].mean(),6)}, {np.round(dlc_output[x].std(),6)}, {np.round(dlc_output[x].std(),6)}")
            
            assert(np.round(dlc_output[x].mean(),3) == 0 and np.round(dlc_output[y].mean(),3)==0)
            assert(np.round(dlc_output[x].std(),3) == 1 and np.round(dlc_output[y].std(),3)==1)


    return dlc_outputs

def get_norm_dict(dlc_outputs):
    norm_dict = {}

    for part in bodyparts:
        x = f"{part}_x"; y = f"{part}_y"
        part_x_all = pd.concat([dlc_output[x] for dlc_output in dlc_outputs])
        part_y_all = pd.concat([dlc_output[y] for dlc_output in dlc_outputs])
        
        part_x_mean = np.mean(part_x_all)
        part_y_mean = np.mean(part_y_all)
        part_x_std = np.std(part_x_all)
        part_y_std = np.std(part_y_all)

        norm_dict.update({f"{x}_mean": part_x_mean, f"{x}_std": part_x_std, f"{y}_mean": part_y_mean, f"{y}_std": part_y_std})

def get_fname_csv(subj_id, sess_id):
    return f"{subj_id}_DynamicForaging_{sess_id}_cam0_00000000DLC_resnet50_MR15Jul8shuffle1_300000.csv"

def get_fname_vid(subj_id, sess_id):
    return f"{subj_id}_DynamicForaging_{sess_id}_cam0_00000000.avi"