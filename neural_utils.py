import numpy as np
import pathlib
from functions import features_change, canton_split
import pandas as pd
from sklearn.model_selection import KFold

def drop_zero_flows(data, drop_ratio=0.5, random_state=37):
    rng = np.random.default_rng(random_state)
    zero_mask = (data[:, -1] == 0)
    
    zero_flow_indices = np.arange(len(data))[zero_mask]
    keep_zero_indices = rng.choice(zero_flow_indices, round(len(zero_flow_indices) * (1-drop_ratio)), replace=False)
    
    final_mask = (~zero_mask)
    final_mask[keep_zero_indices] = True
    return data[final_mask]

column_names = ['commune_1', 'commune_2', 'distance', 'year', 'pop_1', 'pop_2',
    'T_Mann', 'T_Frau', 'Etr_Total', 'Accidents dégâts matériels',
    'Accidents avec dommages corporels', 'Morts', 'BFS_NUMMER',
    'GEM_FLAECH', 'EINWOHNERZ', '0-25', '25-65', '65+', 'canton_code',
    'unemployment', 'gdp', 'T_Mann_2', 'T_Frau_2', 'Etr_Total_2',
    'Accidents dégâts matériels_2', 'Accidents avec dommages corporels_2',
    'Morts_2', 'BFS_NUMMER_2', 'GEM_FLAECH_2', 'EINWOHNERZ_2', '0-25_2',
    '25-65_2', '65+_2', 'canton_code_2', 'unemployment_2', 'gdp_2', 'flow']

logscale_features_const = [
    'distance', 'pop_1', 'pop_2',
    'T_Mann', 'T_Frau', 'Etr_Total', 'Accidents dégâts matériels',
    'Accidents avec dommages corporels', 
    'GEM_FLAECH', 'EINWOHNERZ', '0-25', '25-65', '65+',
    'T_Mann_2', 'T_Frau_2', 'Etr_Total_2',
    'Accidents dégâts matériels_2', 'Accidents avec dommages corporels_2',
    'GEM_FLAECH_2', 'EINWOHNERZ_2', '0-25_2', '25-65_2', '65+_2', 
    "Morts", "Morts_2", "unemployment", "gdp", "unemployment_2", "gdp_2"
] # logscales all columns the model uses

to_remove = ['commune_1', 'commune_2', 'year', 'BFS_NUMMER', 'canton_code','BFS_NUMMER_2', 'canton_code_2']
to_keep = [col for col in column_names if col not in to_remove]
    
def useful_columns_as_numpy(dataframe):
    return dataframe[to_keep].to_numpy()

def prepare_data(classification, zero_drop_ratio = 0.5):
    data_path = pathlib.Path("../data/data_y_v2.npy")
    if classification:
        data = np.load(data_path, mmap_mode="r")
    else:
        full_data = np.load(data_path, mmap_mode="r")
        data = full_data[(full_data[:,-1] > 0), :]

    logscale_features = logscale_features_const.copy() # copy() is important!!
    if not classification:
        logscale_features.append("flow") # regressor works with log(target)

    df = pd.DataFrame(data, columns=column_names)
    df = features_change(df, logscale_features, "log")
    
    # Canton 3 (Luzern) serves as test data.
    train_df, test_df = canton_split(df, canton_ids=[3])
    # Canton 19 as validation (Aargau)
    train_df, val_df = canton_split(train_df, canton_ids=[19])
    
    

    train_data = useful_columns_as_numpy(train_df)
    test_data = useful_columns_as_numpy(test_df)
    val_data = useful_columns_as_numpy(val_df)

    if classification:
        train_data[:, -1] = np.where(train_data[:,-1] == 0, 0, 1) # convert flow values to classes
        test_data[:, -1] = np.where(test_data[:,-1] == 0, 0, 1) # convert flow values to classes
        val_data[:, -1] = np.where(val_data[:,-1] == 0, 0, 1) # convert flow values to classes

    test_df = None
    val_df = None
    train_df = None
    df = None


    

    if classification:
        train_data = drop_zero_flows(train_data, zero_drop_ratio)
    
    return (train_data, val_data, test_data)


# crossvalidation of different models (section [...])
def split_data_respect_years(data, num_splits, random_state=42):
    # distributes the data (2D numpy array) into num_splits (int) different buckets, roughly evenly.
    # Important: assures that all years of the same (commune_id, commune_id2) pair go into the same bucket
    # Returns INDICES of rows to use in each split, not the data itself! (similar to KFold.split)
    pass
    kfolds = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    col_indices = [column_names.index("commune_1"), column_names.index("commune_2")]
    unique_pairs, inverse = np.unique(data[:, col_indices], axis=0, return_inverse=True)
    for train_pairs_index, val_pairs_index in kfolds.split(unique_pairs):
        #print("Hi", train_pairs_index)
        full_train_mask = np.isin(inverse, train_pairs_index)
        full_val_mask = np.isin(inverse, val_pairs_index)
        assert((full_train_mask == ~full_val_mask).all())
        indices = np.arange(len(data))
        yield indices[full_train_mask], indices[full_val_mask]