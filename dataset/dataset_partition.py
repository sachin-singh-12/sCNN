########################################################
# partitioning augmented dataset into 
########################################################
import pandas as pd
from sklearn.model_selection import train_test_split
I = pd.read_csv('augmented_data.csv')
train, val = train_test_split(I, test_size=0.02)
train = train.sort_index(axis=0)
val = val.sort_index(axis=0)
train.to_csv('aug_happei_train.csv', sep=',', index=False)
val.to_csv('aug_happei_val.csv', sep=',',  index=False)

