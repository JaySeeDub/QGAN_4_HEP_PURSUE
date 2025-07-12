import h5py
import numpy as np

input_files = [
    'QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.h5',
    'QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.h5',
    'QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.h5'
]

output_file = 'QCDToGGQQ_IMGjet_combined.h5'

# First, figure out total size per dataset
dataset_shapes = {}
dataset_dtypes = {}
total_lengths = {}

for file in input_files:
    with h5py.File(file, 'r') as f:
        for key in f.keys():
            data_shape = f[key].shape
            data_dtype = f[key].dtype
            n = data_shape[0]
            dataset_shapes[key] = data_shape[1:]  # Exclude first dimension
            dataset_dtypes[key] = data_dtype
            total_lengths[key] = total_lengths.get(key, 0) + n

# Create output datasets
with h5py.File(output_file, 'w') as f_out:
    out_dsets = {}
    for key in dataset_shapes:
        shape = (total_lengths[key],) + dataset_shapes[key]
        out_dsets[key] = f_out.create_dataset(key, shape=shape, dtype=dataset_dtypes[key])

    # Stream data from input files to output file
    for key in dataset_shapes:
        write_index = 0
        for file in input_files:
            with h5py.File(file, 'r') as f_in:
                data = f_in[key]
                n = data.shape[0]
                out_dsets[key][write_index:write_index+n] = data  # copy chunk
                write_index += n

print(f"✅ Finished streaming merge to '{output_file}'")
