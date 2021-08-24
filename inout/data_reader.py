import h5py
import numpy as np


def read_dijet_samples(path):

    ''' read dijet event samples from file 
        path ... path to input file containing jetConstituentList dataset
        returns [2*N x 100 x 3] np.array for 2*N single jets, each with 100 particles, each having 3 features
    '''

    key =  'jetConstituentsList'
    with h5py.File(path,'r') as f:
        consti = np.asarray(f.get(key)) # [N x 2 x 100 x 3]
    samples = np.vstack([consti[:,0,:,:], consti[:,1,:,:]])
    np.random.shuffle(samples)
    return samples

