import numpy as np
import sys, os, subprocess, scikits.audiolab, random, time, glob

PWD = os.path.basename(os.getcwd())
print 'PWD is', PWD
#store dataset name
DATASET_NAME = str(sys.argv[1])
#create the 
print "creating directory for", DATASET_NAME
os.makedirs(DATASET_NAME)
#move samples from directory to use dataset name
print "moving samples"
types = {'wav', "mp3"}
for t in types:
    os.system('mv ./downloads/*.{} ./{}/'.format(t, DATASET_NAME))
#run proprocess
print "preprocessing"
OUTPUT_DIR=os.path.join(DATASET_NAME, "parts")
os.makedirs(OUTPUT_DIR)
print os.path.join(PWD, '/preprocess_file_list.txt')
# Step 1: write all filenames to a list
with open(os.path.join(PWD, '/preprocess_file_list.txt'), 'w') as f:
    for dirpath, dirnames, filenames in os.walk(DATASET_NAME):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith("mp3"):
                f.write("file '" + dirpath + '/'+ filename + "'\n")

# Step 2: concatenate everything into one massive wav file
print "concatenate all files"
os.system('pwd')
os.system("ffmpeg -f concat -safe 0 -i /preprocess_file_list.txt {}/preprocess_all_audio.wav".format(OUTPUT_DIR))
audio = "preprocess_all_audio.wav"
print "get length"
# # get the length of the resulting file
length = float(subprocess.check_output('ffprobe -i {}/{} -show_entries format=duration -v quiet -of csv="p=0"'.format(OUTPUT_DIR, audio), shell=True))
print length, "DURATION"
# reverse the audio file
if sys.argv[2] == True:
    os.system("sox preprocess_all_audio.wav reverse_preprocess_audio.wav reverse")
    audio = "reverse_preprocess_audio.wav"
print "print big file into chunks"
# # Step 3: split the big file into 8-second chunks
for i in xrange((int(length)//8 - 1)/3):
    os.system('ffmpeg -ss {} -t 8 -i {}/{} -ac 1 -ab 16k -ar 16000 {}/p{}.flac'.format(i, OUTPUT_DIR, audio, OUTPUT_DIR, i))
print "clean up"
# # Step 4: clean up temp files
#os.system('rm {}/preprocess_all_audio.wav'.format(OUTPUT_DIR))
os.system('rm {}/preprocess_file_list.txt'.format(OUTPUT_DIR))
print 'save as .npy'
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

paths = sorted(glob.glob(OUTPUT_DIR+"/*.flac"))
__fixed_shuffle(paths)

arr = [(scikits.audiolab.flacread(p)[0]).astype('float16') for p in paths]
np_arr = np.array(arr)
# 88/6/6 split
length = len(np_arr)
train_size = int(np.floor(length * .88)) # train
test_size = int(np.floor(length * .06)) # test

np.save(join(DATASET_NAME,'all_music.npy'), np_arr)
np.save(join(DATASET_NAME,'music_train.npy'), np_arr[:train_size])
np.save(join(DATASET_NAME,'music_valid.npy'), np_arr[train_size:train_size + test_size])
np.save(join(DATASET_NAME,'music_test.npy'), np_arr[train_size + test_size:])

#pass dataset name through two_tier.py || three_tier.py to datasets.py