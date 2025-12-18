import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Noramlize
def normalize_column(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        x_min = df[col].min()
        x_max = df[col].max()
        df[col] = (df[col] - x_min) / (x_max - x_min)
    return df

def normalize_with_train_stats(X_train, X_val, X_test, columns):
    """Normalize all sets using training set min/max"""
    for col in columns:
        # Calculate from TRAIN only
        x_min = X_train[col].min()
        x_max = X_train[col].max()
        
        # Apply same transformation to all
        X_train[col] = (X_train[col] - x_min) / (x_max - x_min + 1e-8)
        X_val[col] = (X_val[col] - x_min) / (x_max - x_min + 1e-8)
        X_test[col] = (X_test[col] - x_min) / (x_max - x_min + 1e-8)
    
    return X_train, X_val, X_test

def feature_interaction(df, column_pairs, interaction):
    possible_interactions = ['multiply', 'ratio', 'difference']
    if interaction not in possible_interactions:
        print('Interaction not in possible interactions')
    for col1, col2 in column_pairs:
        if interaction == 'multiply':
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        elif interaction == 'ratio':
                df[f'{col1}/{col2}_ratio'] = df[col1] / (df[col2] + 1e-5)
        elif interaction == 'difference':
            df[f'{col1}_minus_{col2}'] = df[col1] - (df[col2])
    return df

def gravity(df, pop1, pop2, distance, beta = 2):
    df['gravity_raw'] = (df[pop1] * df[pop2]) / ((df[distance] + 1e-5) ** beta)
    return df

def change_dtype_cat(cat_features, dataframe):
    for idx in cat_features:
        col = dataframe.columns[idx]
        dataframe[col] = dataframe[col].astype(str)
    return dataframe

# Changes features
def features_change(df, columns, method):
    possible_methods = ['log', 'square', 'sqrt', 'logit']
    if method not in possible_methods:
        print(f"I don't do this method. Choose between {possible_methods}")
        return df 
    
    for col in columns:
        if method == 'log':
            df[col] = np.log(df[col] + 1e-5) 
        elif method == 'square':
            df[col] = np.square(df[col])
        elif method == 'sqrt':
            df[col]= np.sqrt(df[col])
        elif method == 'logit':
            epsilon = 1e-5
            df = normalize_column(df, col)
            df[col] = np.clip(df[col], epsilon, 1 - epsilon)
            df[col] = np.log(df[col] / (1 - df[col]))
    return df

def features_engineering(df):
    df = feature_interaction(df, [('pop_1', 'pop_2')], 'multiply')
    print('Column for multiplied population added')
    df = feature_interaction(df, [('pop_1', 'pop_2')], 'ratio')
    print('Column for population ratio added')
    df = feature_interaction(df, [('gdp', 'gdp_2')], 'difference')
    print('Column for difference between gdp of the communes added')
    df = features_change(df, ['distance'], "square") 
    print('Column for distance between the communes squared added')
    df = gravity(df, 'pop_1', 'pop_2', 'distance', beta = 2)
    print('Column for gravitation added')
    return df

# Cleaning a certain data format 
def cleaning_stat_data(data):
    current_canton = None
    canton_list = []
    for val in data['Commune'].astype(str):
        val = val.strip()
        # Canton
        if val.startswith("- "):
            current_canton = val[2:].strip()
            canton_list.append(None)
            continue

        # Districts
        if val.startswith(">> "):
            canton_list.append(None)
            continue

        # Communes
        if val.startswith("...."):
            canton_list.append(current_canton)
            
            continue

        else:
            canton_list.append(None)

    data["canton"] = canton_list
    data = data[
        ~(
            data["Commune"].astype(str).str.startswith(">> ") |
            data["Commune"].astype(str).str.startswith("- ")
        )
    ]

    data[['Code', 'commune_name']] = data['Commune'].str.replace(r'^[.\s]+', '', regex=True)\
                                              .str.extract(r'(\d+)\s+(.+)')

    data = data.drop("Commune", axis=1).reset_index(drop=True)
    return data

# NAN values replacement with regression according to population
def Nan_regression(df, cols_to_predict, col_for_prediction):

    for col in cols_to_predict:
        mask = (
            df[col].notna() &
            df[col_for_prediction].notna() &
            (df[col] > 0) &
            (df[col_for_prediction] > 0)
        )

        if mask.sum() < 2:
            continue

        X = np.log(df.loc[mask, col_for_prediction]).values.reshape(-1, 1)
        y = np.log(df.loc[mask, col])

        model = LinearRegression()
        model.fit(X, y)

        missing = (
            df[col].isna() &
            (df[col_for_prediction] > 0)
        )

        if missing.sum() == 0:
            continue

        X_missing = np.log(df.loc[missing, col_for_prediction]).values.reshape(-1, 1)
        df.loc[missing, col] = np.exp(model.predict(X_missing))

    return df


def canton_split(df, canton_ids=[2]):
    """
    Strict split: All flows whose origin OR destination is in the specified canton → test set
    """
    print(f"Splitting with canton {canton_ids} as test set:")
    # test: flows are either thein origin or their destination in the chosen canton 
    # train: flows where neither origin or destination is in chosen canton
    test_mask = (df['canton_code'].isin(canton_ids)) | (df['canton_code_2'].isin(canton_ids)) # TODO check this line
    train_mask = ~test_mask
    
    # splitting the data
    test_df = df[test_mask]
    train_df = df[train_mask]

    # we verify no flows involving the test canton are in train set
    train_has_test_canton = train_df[
        (train_df['canton_code'].isin(canton_ids)) | 
        (train_df['canton_code_2'].isin(canton_ids))
    ]
    
    print(f"Total flows: {len(df):,}")
    print(f"Train size: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test size: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

    
    assert len(train_has_test_canton) == 0, \
        f"{len(train_has_test_canton)} flows involving cantons {canton_ids} found in train set!"

    return train_df, test_df


def class_imbalance_check(train_df, test_df):
    """
    Check for class imbalance in train and test sets after canton split.
    """
    print(f"Class Imbalance Check :")
    train_zero = (train_df['flow'] == 0).sum()
    test_zero = (test_df['flow'] == 0).sum()
    print(f"Zero flows in train set: {train_zero} ({train_zero/len(train_df)*100:.2f}%)")
    print(f"Zero flows in test set: {test_zero} ({test_zero/len(test_df)*100:.2f}%)")




def handle_class_imbalance(df_train, zero_drop_ratio=0.2, random_state=37):
    """ Drop a specified percentage of zero flows.

        Parameters:
        -----------
        df_train : DataFrame
            Training data with 'flow' column
        zero_drop_ratio : float (0-1)
            Percentage of zero flows to drop (e.g., 0.5 = drop 50% of zeros)
        random_state : int
            Random seed for reproducibility """
    np.random.seed(random_state)
    
    zero_mask = df_train['flow'] == 0
    non_zero_mask = df_train['flow'] > 0
    
    zero_df = df_train[zero_mask]
    
    non_zero_df = df_train[non_zero_mask]
    
    # how many zeros to keep
    n_zeros_keep = int(len(zero_df) * (1 - zero_drop_ratio))
    
    print(f"Dropping {zero_drop_ratio*100:.0f}% of zeros in training data")
    
    # Sample zero flows
    zero_sample = zero_df.sample(n=n_zeros_keep, random_state=random_state)
    
    # Combine and shuffle
    balanced_df = pd.concat([non_zero_df, zero_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df


def classification_fct(y_data, threshold=0):
    print("Flow reclassified as 1: flow present, 0: no flow")
    return (y_data > threshold).astype(int)


# Final function to build train and test sets: 
TO_REMOVE = ['commune_1', 'commune_2', 'BFS_NUMMER', 'canton_code','BFS_NUMMER_2', 'canton_code_2', 'flow']



def build_train_test_val(
    dataframe,
    test_canton_ids=[3],
    val_canton_ids=[19],
    zero_drop_ratio=0.2,
    random_state=37,
    features=None,
    classify=False):
    """ Final function to build train, validation and test sets 
    Args:
        dataframe: full dataframe
        test_canton_ids: list of canton IDs to use as test set
        val_canton_ids: list of canton IDs to use as validation set
        zero_drop_ratio: percentage of zero flows to drop from training set
        random_state: random seed for reproducibility
        to_remove: list of columns to remove from features (for example IDs and years)
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
       """

    if classify == False:
        dataframe = dataframe[dataframe['flow'] > 0]

    if features is None:
        features = TO_REMOVE

    df = dataframe.copy()

    # One-hot encode year
    for year_val in sorted(df['year'].unique()):
        df[f'year_{year_val}'] = (df['year'] == year_val).astype(int)

    # Drop original year
    features = features + ['year']

    # Features to keep
    to_keep = [col for col in df.columns if col not in features]

    # Spatial split
    train_df, test_df = canton_split(df, canton_ids=test_canton_ids)
    train_df, val_df = canton_split(train_df, canton_ids=val_canton_ids)

    # Handle class imbalance (training only)
    train_df = handle_class_imbalance(train_df, zero_drop_ratio, random_state)

    X_train = train_df[to_keep].copy()
    X_val = val_df[to_keep].copy()
    X_test = test_df[to_keep].copy()

    # Targets
    if classify == True:
        y_train = classification_fct(train_df['flow'])
        y_val = classification_fct(val_df['flow'])
        y_test = classification_fct(test_df['flow'])
    else:
        y_train = train_df['flow']
        y_val = val_df['flow']
        y_test = test_df['flow']

    return X_train, y_train, X_val, y_val, X_test, y_test

def binary_exp_cantons(X_train, X_test, X_val):
    X_train = pd.get_dummies(
        X_train,
        columns=["canton_code", "canton_code_2"],
        prefix=["canton_1", "canton_2"],
        drop_first=False
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["canton_code", "canton_code_2"],
        prefix=["canton_1", "canton_2"],
        drop_first=False
    )

    X_val = pd.get_dummies(
        X_val,
        columns=["canton_code", "canton_code_2"],
        prefix=["canton_1", "canton_2"],
        drop_first=False
    )
    return X_train, X_test, X_val
