#!/usr/bin/env python3
import numpy as np
import random

from tinygrad.state import get_parameters
from tinygrad.nn.optim import Adam
# from extra.training import train, evaluate
from tinygrad.tensor import Tensor

import dataset as ds
import utils
# from torch.utils.data import DataLoader
# import torch
import datetime
from transformer_timeseries import Transformer
import numpy as np

# Hyperparams
test_size = 0.2
batch_size = 128 # The batch size affects some indicators such as overall training time, training time per epoch, quality of the model, and similar. Usually, we chose the batch size as a power of two, in the range between 16 and 512. But generally, the size of 32 is a rule of thumb and a good initial choice.
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1)

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 8 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# * Read data
data = utils.read_data('dfs_merged_upload')

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=Tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making dataloader
# training_data = DataLoader(training_data, batch_size)
training_data = utils.create_batches(training_data, batch_size)

i, batch = next(enumerate(training_data))

src, trg, trg_y = batch

# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
if batch_first == False:
    print("Shape of src before transpose:", src.shape)
    print("Shape of trg before transpose:", trg.shape)

    shape_before = src.shape
    src = np.transpose(src, (1, 0, 2))
    print("src shape changed from {} to {}".format(shape_before, src.shape))

    shape_before = trg.shape
    trg = np.transpose(trg, (1, 0, 2))
    print("trg shape changed from {} to {}".format(shape_before, trg.shape))


# -----------------------------------

# if __name__ == "__main__":
#   model = Transformer(10, 6, 2, 128, 4, 32)
#   X_train, Y_train, X_test, Y_test = make_dataset()
#   lr = 0.003
#   for i in range(10):
#     optim = Adam(get_parameters(model), lr=lr)
#     utils.train(model, X_train, Y_train, optim, 50, BS=64)
#     acc, Y_test_preds = utils.evaluate(model, X_test, Y_test, num_classes=10, return_predict=True)
#     lr /= 1.2
#     print(f'reducing lr to {lr:.4f}')
#     if acc > 0.998:
#       wrong=0
#       for k in range(len(Y_test_preds)):
#         if (Y_test_preds[k] != Y_test[k]).any():
#           wrong+=1
#           a,b,c,x = X_test[k,:2], X_test[k,2:4], Y_test[k,-3:], Y_test_preds[k,-3:]
#           print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
#       print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')

# if __name__ == "__main__":
#   model = Transformer(input_size, 6, 2, 128, 4, 32)
#   lr = 0.003
#   for i in range(10):
#     optim = Adam(get_parameters(model), lr=lr)
#     utils.train(model, src, trg, optim, 50, BS=64)
#     acc, Y_test_preds = utils.evaluate(model, src, trg, num_classes=input_size, return_predict=True)
#     lr /= 1.2
#     print(f'reducing lr to {lr:.4f}')
#     if acc > 0.998:
#       wrong=0
#       for k in range(len(Y_test_preds)):
#         if (Y_test_preds[k] != trg[k]).any():
#           wrong+=1
#           a,b,c,x = src[k,:2], src[k,2:4], trg[k,-3:], Y_test_preds[k,-3:]
#           print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
#       print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')

if __name__ == "__main__":
    # Obtain the training and test data
    X_train, y_train, X_test, y_test = utils.get_dataset('data/dfs_merged_upload.csv', batch_size=batch_size, test_size=test_size)

    model = Transformer(input_size, max_seq_len, n_encoder_layers, dim_val, n_heads, step_size)
    lr = 0.003
    for i in range(10):
        optim = Adam(get_parameters(model), lr=lr)
        utils.train(model, X_train, y_train, optim, 50, BS=64)
        acc, y_test_preds = utils.evaluate(model, X_test, y_test, num_classes=input_size, return_predict=True)
        lr /= 1.2
        print(f'reducing lr to {lr:.4f}')
        if acc > 0.998:
            wrong=0
            for k in range(len(y_test_preds)):
                if (y_test_preds[k] != y_test[k]).any():
                    wrong+=1
                    a,b,c,x = X_test[k,:2], X_test[k,2:4], y_test[k,-3:], y_test_preds[k,-3:]
                    print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
            print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')
