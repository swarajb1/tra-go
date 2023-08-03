import h5py

filename = "./training/model_weights.h5"

h5 = h5py.File(filename, "r")

print(h5["top_level_model_weights"])


h5.close()
