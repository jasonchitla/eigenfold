import pandas as pd
import numpy as np

def preprocess():
    df = pd.read_csv('./pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]; del df['saved']
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df['split'] = np.where(df.release_date < '2022-01-01', 'train', 'val')
    val_mask = df['split'] == 'val'
    num_test = int(0.3 * sum(val_mask))
    val_indices = df[val_mask].index.to_numpy()
    # Change the last num_test number of the 'val' rows to 'test'
    df.loc[val_indices[-num_test:], 'split'] = 'test'
    train_count = (df['split'] == 'train').sum()
    val_count = (df['split'] == 'val').sum()
    test_count = (df['split'] == 'test').sum()
    print(f"train count: {train_count}")
    print(f"val count: {val_count}")
    print(f"test count: {test_count}")
    df.to_csv('preprocessed.csv')
    
if __name__ == "__main__":
    preprocess()