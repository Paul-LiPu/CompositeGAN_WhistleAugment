import numpy as np
import os
import h5py
import os.path

def get_parentdir(dir):
    parent_dir = os.path.dirname(dir)
    parent_folder = os.path.basename(parent_dir)
    return parent_folder

def classify_filelist(file_list):
    classes = map(get_parentdir, file_list)
    file_dict = {}
    for i in range(0, len(file_list)):
        if not classes[i] in file_dict.keys():
            file_dict[classes[i]] = []
        file_dict[classes[i]].append(file_list[i])
    return file_dict

def read_h5_pos(file, pos, nsamples):
    h5file = h5py.File(file, 'r')
    data = h5file['data'][pos:pos+nsamples]
    label = h5file['label'][pos:pos+nsamples]
    h5file.close()
    return data, label


def read_h5_pos2(data0, label0, pos, nsamples):
    data = data0[pos:pos+nsamples]
    label = label0[pos:pos+nsamples]
    return data, label


def read_h5_length(file):
    # print os.path.exists(file)
    h5file = h5py.File(file)
    length = len(h5file['data'])
    h5file.close()
    return length


class HDF5_Dataset():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        length_list = list(map(read_h5_length, hdf5_list))
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label

    def __next__(self):
        return self.next()



class HDF5_Dataset_in_RAM():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr = 0

        data = []
        label = []
        for file in hdf5_list:
            h5file = h5py.File(file, 'r')
            curr_data = h5file['data'][...]
            curr_label = h5file['label'][...]
            h5file.close()
            data.append(curr_data)
            label.append(curr_label)
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        self.total_count = data.shape[0]
        
        self.data = data
        self.label = label

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        data, label = self.data[self.curr:self.curr+self.batch_size, ], self.label[self.curr:self.curr+self.batch_size, ]
        self.curr += self.batch_size
        if self.curr >= self.total_count:
            self.curr = 0
        return data, label

    def shuffle(self):
        idx = np.random.permutation(self.total_count)
        self.data = self.data[idx, ]
        self.label = self.label[idx, ]
        
        
    def __next__(self):
        return self.next()


class HDF5_Dataset_transpose():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        length_list = list(map(read_h5_length, hdf5_list))
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        data = np.transpose(data, (0, 1, 3, 2))
        label = np.transpose(label, (0, 1, 3, 2))
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label

    def __next__(self):
        return self.next()
