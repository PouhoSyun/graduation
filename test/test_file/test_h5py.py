import h5py

file = h5py.File('test/test_file/rec1501902136.hdf5', 'r+')
print(file)
file.close()
