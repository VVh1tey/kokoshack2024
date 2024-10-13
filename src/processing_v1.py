
import numpy as np
import pandas as pd
from datetime import timedelta

def load_data(train_actions_path, stokman_catalog_path):
    train_actions = pd.read_parquet(train_actions_path, engine='pyarrow')
    train_actions = train_actions.explode('products')
    stokman_catalog = pd.read_parquet(stokman_catalog_path, engine='pyarrow')
    
    return train_actions, stokman_catalog

def preprocess_data(train_actions):
    train_actions['date'] = pd.to_datetime(train_actions['date'])
    return train_actions

def calculate_user_features(train_actions, stokman_catalog):
    ACTIONS = {
        0: 'view',
        1: 'like',
        2: 'addB',
        3: 'delB',
        4: 'clearB',
        5: 'order',
        6: 'listB',
        7: 'visit',
        8: 'visitCategory',
        9: 'search'
    }

    user_features = train_actions.groupby('user_id').agg(
        total_actions=('action', 'count'),
        nunique_products_number=('productId', pd.Series.nunique),
    ).reset_index()

    for action_code, action_name in ACTIONS.items():
        action_col = f"{action_name}_number"
        user_features[action_col] = train_actions[train_actions['action'] == action_code].groupby('user_id')['action'].count().fillna(0).values

    return user_features

def save_user_features(user_features, path):
    user_features.to_parquet(path, index=False)

