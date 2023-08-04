import numpy as np
import pandas as pd
from tqdm import trange
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import getenv
from typing import Optional, Any, Union, Callable, Tuple
from pathlib import Path

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy,
        transform=lambda x: x, target_transform=lambda x: x, noloss=False):
  Tensor.training = True
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=getenv('CI', False))):
    print('X_train.shape', X_train.shape)
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(transform(X_train[samp]), requires_grad=False)
    y = target_transform(Y_train[samp])

    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)

    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()

    # printing
    if not noloss:
      cat = np.argmax(out.cpu().numpy(), axis=-1)
      accuracy = (cat == y).mean()

      loss = loss.detach().cpu().numpy()
      losses.append(loss)
      accuracies.append(accuracy)
      t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]


def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
  Tensor.training = False
  def numpy_eval(Y_test, num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange((len(Y_test)-1)//BS+1, disable=getenv('CI', False)):
      x = Tensor(transform(X_test[i*BS:(i+1)*BS]))
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.cpu().numpy()
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean(), Y_test_preds

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  acc, Y_test_pred = numpy_eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc

def read_data(filename: Union[str, Path] = "data") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        filename: str or Path object specifying the path to the directory
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index
                            containing the timestamps
    """




    # Ensure that `data_dir` is a Path object
    data_dir = Path("data")

    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))

    # Check if no csv files
    if len(csv_files) <= 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    # If filename, use as csv file, else use first
    data_path = csv_files[0]
    if len(filename) > 0:
        data_path = Path(f"data/{filename}.csv")

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path,
        # parse_dates=[timestamp_col_name],
        # index_col=[timestamp_col_name],
        # infer_datetime_format=True,
        low_memory=False
    )

    # Make sure all "n/e" values have been removed from df.
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    # data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """

    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns

    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')

    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences.

        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences.

        Args:
            num_obs (int): Number of observations (time steps) in the entire
                           dataset for which indices must be generated, e.g.
                           len(data)

            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50
                               time steps, window_size = 100+50 = 150

            step_size (int): Size of each step as the data sequence is traversed
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size],
                             and the next will be [1:window_size].

        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing

        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0

        subseq_last_idx = window_size

        indices = []

        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))

            subseq_first_idx += step_size

            subseq_last_idx += step_size

        return indices

# def create_batches(data, batch_size):
#     # Converting to numpy array if not already
#     if type(data) is not np.ndarray:
#         data = np.array(data)

#     # Making sure we have at least 1 split
#     num_splits = max(1, len(data) // batch_size)

#     # Splitting data into batches
#     batches = np.array_split(data, num_splits)
#     return batches
# def create_batches(data, batch_size):
#     n = len(data)
#     res = []
#     for i in range(0, n, batch_size):
#         batch = data[i:i+batch_size]
#         src, trg, trg_y = list(zip(*[(d.enc_input, d.dec_input, d.target) for d in batch]))
#         res.append((np.array(src), np.array(trg), np.array(trg_y)))
#     return res

# def create_batches(data, batch_size):
#     n = len(data)
#     batches = []
#     for i in range(0, n, batch_size):
#         batch_indices = range(i, min(i + batch_size, n))
#         batch = [data[idx] for idx in batch_indices]
#         src, trg, trg_y = list(zip(*batch))

#         # Convert to numpy arrays and add an extra dimension
#         src = np.expand_dims(np.array(src), axis=-1)
#         trg = np.expand_dims(np.array(trg), axis=-1)
#         trg_y = np.expand_dims(np.array(trg_y), axis=-1)

#         print('src.shape', src.shape)
#         print('len(src.shape)', len(src.shape))

#         # Verify shapes
#         if len(src.shape) != 3:
#             raise ValueError(f"Unexpected shape for src: {src.shape}. "
#                              f"Expected shape: (batch_size, sequence_length, num_features)")
#         if len(trg.shape) != 3:
#             raise ValueError(f"Unexpected shape for trg: {trg.shape}. "
#                              f"Expected shape: (batch_size, sequence_length, num_features)")
#         if len(trg_y.shape) != 3:
#             raise ValueError(f"Unexpected shape for trg_y: {trg_y.shape}. "
#                              f"Expected shape: (batch_size, sequence_length, num_features)")

#         batches.append((src, trg, trg_y))

#     return batches
def create_batches(data, batch_size):
    n = len(data)
    batches = []
    for i in range(0, n, batch_size):
        batch_indices = range(i, min(i + batch_size, n))
        batch = [data[idx] for idx in batch_indices]
        src, trg, trg_y = list(zip(*batch))

        # Convert to numpy arrays and make sure they are at least two-dimensional
        src = np.array(src)
        trg = np.array(trg)
        trg_y = np.array(trg_y)
        if src.ndim == 1:
            src = src[:, np.newaxis]
        if trg.ndim == 1:
            trg = trg[:, np.newaxis]
        if trg_y.ndim == 1:
            trg_y = trg_y[:, np.newaxis]

        # Add an additional dimension if only one feature
        if len(src.shape) == 2:
            src = np.expand_dims(src, axis=-1)
        if len(trg.shape) == 2:
            trg = np.expand_dims(trg, axis=-1)
        if len(trg_y.shape) == 2:
            trg_y = np.expand_dims(trg_y, axis=-1)

        print('src.shape', src.shape)
        print('len(src.shape)', len(src.shape))

        # Verify shapes
        if len(src.shape) != 3:
            raise ValueError(f"Unexpected shape for src: {src.shape}. "
                             f"Expected shape: (batch_size, sequence_length, num_features)")
        if len(trg.shape) != 3:
            raise ValueError(f"Unexpected shape for trg: {trg.shape}. "
                             f"Expected shape: (batch_size, sequence_length, num_features)")
        if len(trg_y.shape) != 3:
            raise ValueError(f"Unexpected shape for trg_y: {trg_y.shape}. "
                             f"Expected shape: (batch_size, sequence_length, num_features)")

        batches.append((src, trg, trg_y))

    return batches


# ---------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Convert datetime to Unix timestamp
    data['timestamp'] = (data['timestamp'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')

    # Sort the data by timestamp
    data.sort_values('timestamp', inplace=True)

    return data

# def generate_batches(data, batch_size):
#     # Split data into X and y
#     X = data.drop('FCR_N_PriceEUR', axis=1).values
#     y = data['FCR_N_PriceEUR'].values

#     # Reshape X and y to be divisible by batch_size
#     X = X[:-(X.shape[0] % batch_size)]
#     y = y[:-(y.shape[0] % batch_size)]

#     # Reshape to (num_batches, batch_size, num_features)
#     X = X.reshape(-1, batch_size, X.shape[1])
#     y = y.reshape(-1, batch_size, 1)

#     return X, y
def generate_batches(data, batch_size):
    # Split data into X and y
    X = data.drop('FCR_N_PriceEUR', axis=1).values
    y = data['FCR_N_PriceEUR'].values

    # Reshape X and y to be divisible by batch_size
    X = X[:-(X.shape[0] % batch_size)]
    y = y[:-(y.shape[0] % batch_size)]

    # Reshape to (num_batches*batch_size, num_features)
    # Here, num_features will act as sequence_length
    X = X.reshape(-1, X.shape[1])
    y = y.reshape(-1, 1)

    return X, y


def get_dataset(filepath, batch_size, test_size):
    # Load the data
    data = load_data(filepath)

    # Split into training and test data
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    # Check that the data was loaded and split correctly
    assert len(train_data) > 0, "Training data is empty."
    assert len(test_data) > 0, "Test data is empty."

    # Generate batches
    X_train, y_train = generate_batches(train_data, batch_size)
    X_test, y_test = generate_batches(test_data, batch_size)

    # # Ensure all data is numerical
    # X_train = X_train.astype(float).values
    # y_train = y_train.astype(float).values
    # X_test = X_test.astype(float).values
    # y_test = y_test.astype(float).values

    return X_train, y_train, X_test, y_test

