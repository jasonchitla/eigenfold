import pandas as pd
import numpy as np

def preprocess():
    df = pd.read_csv('./pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]; del df['saved']
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df = df[df.release_date < '2020-12-01']
    df['split'] = np.where(df.release_date < '2020-05-01', 'train', 'val')
    # Randomly shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # Define the split points
    train_idx = int(0.8 * len(df))
    val_idx = int(0.9 * len(df))  # This is also equal to train_idx + 0.1 * len(df)

    # Assign 'train', 'val', and 'test'
    df['split'] = 'train'
    df.loc[train_idx:val_idx, 'split'] = 'val'
    df.loc[val_idx:, 'split'] = 'test'

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