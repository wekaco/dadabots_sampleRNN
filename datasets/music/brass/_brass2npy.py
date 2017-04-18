import numpy
np = numpy
import scipy.io.wavfile
import scikits.audiolab

import random
import time
import os
import glob

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

data_paths = [
    (os.path.abspath('./fr-parts')),
    (os.path.abspath('./br-parts')),
    (os.path.abspath('./f-parts')),
    (os.path.abspath('./b-parts'))
]

for dp in data_paths:
    paths = sorted(glob.glob(dp+"/*.flac"))
    __fixed_shuffle(paths)
    arr = [(scikits.audiolab.flacread(p)[0]).astype('float16') for p in paths]
    np_arr = np.array(arr)
    print np_arr.shape
    """ BETHOVEEN MUSIC SPLIT
    np.save('all_music.npy', np_arr)
    np.save('music_train.npy', np_arr[:-2*256])
    np.save('music_valid.npy', np_arr[-2*256:-256])
    np.save('music_test.npy', np_arr[-256:])
    """
    # 88/6/6 split
    length = len(np_arr)
    train_size = int(np.floor(length * .88)) # train
    test_size = int(np.floor(length * .06)) # test
    np.save(dp+'all_brass.npy', np_arr)
    np.save(dp+'brass_train.npy', np_arr[:train_size])
    np.save(dp+'brass_valid.npy', np_arr[train_size:train_size + test_size])
    np.save(dp+'brass_test.npy', np_arr[train_size + test_size:])
