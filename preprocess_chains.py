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
    df.to_csv('limit256.csv')
    
if __name__ == "__main__":
    preprocess()