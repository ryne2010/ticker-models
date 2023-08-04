import os
# import torch
from tinygrad.tensor import Tensor
# from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple

class TransformerDataset(pd.DataFrame):
    """
    Dataset class used for transformer models.

    """
    def __init__(self,
        data: Tensor,
        indices: list,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()

        self.indices = indices

        self.data = data

        # print("From get_src_trg: data size = {}".format(data.size()))
        print("From get_src_trg: data size = {}".format(data))

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        #print("From __getitem__: sequence length = {}".format(sequence.shape[0]))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        print(f"src shape: {src.shape}, trg shape: {trg.shape}, trg_y shape: {trg_y.shape}")
        return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: Tensor,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int
        ) -> Tuple[Tensor, Tensor, Tensor]:
        # print('sequence before slicing', sequence)

        assert sequence.shape[0] == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len, :]

        # decoder input
        trg = sequence[enc_seq_len-1:sequence.shape[0]-1, :]

        assert trg.shape[0] == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence
        trg_y = sequence[-target_seq_len:, :]

        assert trg_y.shape[0] == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(-1)
