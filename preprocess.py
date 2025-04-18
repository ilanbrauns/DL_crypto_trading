import pickle 
import numpy as np 
import pandas as pd 

def unpickle(file):
    dct = pd.read_csv(file)

    return dct 