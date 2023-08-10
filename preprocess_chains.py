import pandas as pd
import numpy as np

def preprocess():
    df = pd.read_csv('./pdb_chains.csv', index_col='name')
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]; del df['saved']
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df = df[df.release_date < '2020-12-01']
    df['split'] = np.where(df.release_date < '2019-12-01', 'train', 'val')
    train_count = (df['split'] == 'train').sum()
    val_count = (df['split'] == 'val').sum()
    print(f"train count: {train_count}")
    print(f"val count: {val_count}")
    df.to_csv('preprocessed.csv')
    
if __name__ == "__main__":
    preprocess()