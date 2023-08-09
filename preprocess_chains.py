import pandas as pd
import numpy as np

def preprocess():
    df = pd.read_csv('./pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]; del df['saved']
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df = df[df.release_date < '2020-12-01']
    # Randomly shuffle the dataframe without resetting index
    shuffled_df = df.sample(frac=1)

    # Determine split indices based on the desired percentages
    train_idx = int(0.8 * len(df))
    val_idx = int(0.9 * len(df))

    # Get the shuffled indices
    shuffled_indices = shuffled_df.index.tolist()

    # Assign 'train', 'val', and 'test' based on the shuffled indices
    df.loc[shuffled_indices[:train_idx], 'split'] = 'train'
    df.loc[shuffled_indices[train_idx:val_idx], 'split'] = 'val'
    df.loc[shuffled_indices[val_idx:], 'split'] = 'test'

    # Count the number of 'train', 'val', and 'test' entries
    train_count = (df['split'] == 'train').sum()
    val_count = (df['split'] == 'val').sum()
    test_count = (df['split'] == 'test').sum()

    print(f"Train count: {train_count}")
    print(f"Validation count: {val_count}")
    print(f"Test count: {test_count}")
    df.to_csv('limit256.csv')
    
if __name__ == "__main__":
    preprocess()