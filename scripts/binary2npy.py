import struct
import numpy as np

data_folder_path = "/media/joyful/HDD/media_eval/LIRIS-ACCEDE-data"

def binary2npy(src_file_name, dst_file_name):
    binary = open(src_file_name+'.fc7-1', 'rb')

    s = 5
    size = 1 # will be 4096

    for i in range(s):
        s = struct.unpack('i', binary.read(4))
        size = size * s[0]     

    feature = np.zeros(size)

    for i in range(size):
        val = struct.unpack('f', binary.read(4))
        feature[i] = val[0]    
        

    print dst_file_name, len(feature)
    np.save(dst_file_name, feature)

f = open('media_eval.txt','r')
with open('media_eval.txt','r') as f:
    lines = f.read().splitlines()

print len(lines)

for file_name in lines:
    binary_file_path = data_folder_path + "/output/" + file_name
    
    npy_file_path = data_folder_path + "/output_fc7-1/" + file_name
    
    binary2npy(binary_file_path, npy_file_path)