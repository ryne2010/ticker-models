import numpy as np
from tinygrad.tensor import Tensor

# This class is used to compute the positional encodings
class PositionalEncoding:
    def __init__(self, embedding_dimension):
        # Store the dimension size
        self.embedding_dimension = embedding_dimension

    def __call__(self, sequence_length):
        # Compute the angles
        angles = self.get_angles(np.arange(sequence_length)[:, np.newaxis],
                                  np.arange(self.embedding_dimension)[np.newaxis, :],
                                  self.embedding_dimension)
        # Convert the angles to a Tensor
        positional_encoding = Tensor(angles, requires_grad=False)
        return positional_encoding

    def get_angles(self, position, index, embedding_dimension):
        # This function computes the positional encodings
        angle_rates = 1 / np.power(10000, (2 * (index//2)) / np.float32(embedding_dimension))
        return position * angle_rates

# The TransformerBlock class defines a single block of the transformer architecture.
# It encapsulates the self-attention mechanism and the position-wise feed-forward network.
class TransformerBlock:
  def __init__(self, embedding_dimension, num_heads, feedforward_dimension, prenormalization=False, activation_function=lambda x: x.relu(), dropout_rate=0.1):
    # Checks if the embedding dimension is divisible by the number of heads
    assert embedding_dimension % num_heads == 0, "Embedding dimension must be divisible by number of heads"

    # Number of heads for multi-head attention
    self.num_heads = num_heads
    # Size of each head
    self.head_size = embedding_dimension // num_heads
    # Whether to apply layer normalization before (True) or after (False) the self-attention and feed-forward networks
    self.prenormalization, self.activation_function = prenormalization, activation_function
    # Dropout rate
    self.dropout_rate = dropout_rate

    # Query, Key, Value weights and biases for multi-head self-attention
    self.query = (Tensor.scaled_uniform(embedding_dimension, embedding_dimension), Tensor.zeros(embedding_dimension))
    self.key = (Tensor.scaled_uniform(embedding_dimension, embedding_dimension), Tensor.zeros(embedding_dimension))
    self.value = (Tensor.scaled_uniform(embedding_dimension, embedding_dimension), Tensor.zeros(embedding_dimension))

    # Output projection weights and bias
    self.out_projection = (Tensor.scaled_uniform(embedding_dimension, embedding_dimension), Tensor.zeros(embedding_dimension))

    # Feed-forward network weights and biases
    self.feedforward_1 = (Tensor.scaled_uniform(embedding_dimension, feedforward_dimension), Tensor.zeros(feedforward_dimension))
    self.feedforward_2 = (Tensor.scaled_uniform(feedforward_dimension, embedding_dimension), Tensor.zeros(embedding_dimension))

    # Layer normalization parameters
    self.layer_norm_1 = (Tensor.ones(embedding_dimension), Tensor.zeros(embedding_dimension))
    self.layer_norm_2 = (Tensor.ones(embedding_dimension), Tensor.zeros(embedding_dimension))

  # This method implements the self-attention mechanism
  def attention(self, input_sequence):
    # input_sequence: (batch_size, sequence_length, embedding_dimension)
    # -> (batch_size, sequence_length, embedding_dimension)
    query, key, value = [input_sequence.linear(*param) \
      .reshape(shape=(input_sequence.shape[0], -1, self.num_heads, self.head_size)) \
      for param in [self.query, self.key, self.value]]

    # Permute dimensions for calculation of attention scores
    query = query.permute(order=(0,2,1,3))  # (batch_size, num_heads, sequence_length, head_size)
    key = key.permute(order=(0,2,3,1))      # (batch_size, num_heads, head_size, sequence_length)
    value = value.permute(order=(0,2,1,3))  # (batch_size, num_heads, sequence_length, head_size)

    # Calculate attention scores
    attention_scores = query.dot(key) * (1 / np.sqrt(self.head_size))
    # Normalize scores to probabilities
    attention_weights = attention_scores.softmax() # (batch_size, num_heads, sequence_length, sequence_length)
    # Calculate attention output
    attention_output = attention_weights.dot(value).permute(order=(0,2,1,3)) # (batch_size, sequence_length, num_heads, head_size)

    # Reshape and apply output projection
    return attention_output.reshape(shape=(input_sequence.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out_projection)

  # The call method is where the input goes through the transformer block
  def __call__(self, input_sequence):
    if self.prenormalization:
      input_sequence = input_sequence + self.attention(input_sequence.layernorm().linear(*self.layer_norm_1)).dropout(self.dropout_rate)
      input_sequence = input_sequence + self.activation_function(input_sequence.layernorm().linear(*self.layer_norm_2).linear(*self.feedforward_1)).linear(*self.feedforward_2).dropout(self.dropout_rate)
    else:
      input_sequence = input_sequence + self.attention(input_sequence).dropout(self.dropout_rate)
      input_sequence = input_sequence.layernorm().linear(*self.layer_norm_1)
      input_sequence = input_sequence + self.activation_function(input_sequence.linear(*self.feedforward_1)).linear(*self.feedforward_2).dropout(self.dropout_rate)
      input_sequence = input_sequence.layernorm().linear(*self.layer_norm_2)
    return input_sequence


class Transformer:
  def __init__(self, num_symbols, max_sequence_length, num_layers, embedding_dimension, num_heads, feedforward_dimension):
    self.max_sequence_length, self.num_symbols = max_sequence_length, num_symbols
    # Randomly initialize the embedding weights
    self.embedding = Tensor.scaled_uniform(max_sequence_length + num_symbols, embedding_dimension, requires_grad=False)
    # Initialize the positional encoder
    self.positional_encoder = PositionalEncoding(embedding_dimension)
    self.transformer_blocks = []
    for _ in range(num_layers):
      self.transformer_blocks.append(TransformerBlock(embedding_dimension, num_heads, feedforward_dimension))
    self.final_projection = Tensor.scaled_uniform(embedding_dimension, num_symbols)

  def forward(self, input_sequence):
    batch_size = input_sequence.shape[0]
    input_as_numpy = input_sequence.cpu().numpy().astype(np.int32)
    one_hot_encoding = np.zeros((batch_size, input_sequence.shape[1], self.num_symbols), dtype=np.float32)
    print(f"num_symbols: {self.num_symbols}")
    print(f"input_sequence: {input_sequence}")
    for index in range(input_sequence.shape[1]):
      print(f"input_as_numpy[:, index]: {input_as_numpy[:, index]}")
      one_hot_encoding[range(batch_size), index, input_as_numpy[:, index]] = 1
    one_hot_encoding = one_hot_encoding.reshape(batch_size*input_sequence.shape[1], self.num_symbols)

    # Apply the embedding to the input
    embedded_input_sequence = Tensor(one_hot_encoding, device=input_sequence.device).dot(self.embedding).reshape(shape=(batch_size, input_sequence.shape[1], -1))
    # Add the positional encoding to the embedded input
    input_with_position = embedded_input_sequence + self.positional_encoder(embedded_input_sequence.shape[1])
    # Pass the input through the transformer blocks
    output_sequence = input_with_position.sequential(self.transformer_blocks)
    # Apply the final linear layer and softmax
    logits = output_sequence.reshape(shape=(-1, output_sequence.shape[-1])).dot(self.final_projection).log_softmax()
    # Reshape the output
    output_sequence = logits.reshape(shape=(batch_size, -1, logits.shape[-1]))
    return output_sequence


  # def forward(self, input_sequence):
  #   batch_size = input_sequence.shape[0]
  #   input_as_numpy = input_sequence.cpu().numpy().astype(np.int32)
  #   one_hot_encoding = np.zeros((batch_size, input_sequence.shape[1], self.max_sequence_length + self.num_symbols), dtype=np.float32)
  #   # for index in range(input_sequence.shape[1]):
  #   #   one_hot_encoding[range(batch_size), index, index] = 1
  #   #   one_hot_encoding[range(batch_size), index, self.max_sequence_length + input_as_numpy[:, index]] = 1
  #   for index in range(input_sequence.shape[1]):
  #     print(f"one_hot_encoding shape: {one_hot_encoding.shape}")
  #     print(f"range(batch_size) shape: {len(range(batch_size))}")
  #     print(f"index: {index}")
  #     print(f"self.max_sequence_length + input_as_numpy[:, index] shape: {(self.max_sequence_length + input_as_numpy[:, index]).shape}")
  #     one_hot_encoding[range(batch_size), index, self.max_sequence_length + input_as_numpy[:, index]] = 1
  #   one_hot_encoding = one_hot_encoding.reshape(batch_size*input_sequence.shape[1], self.max_sequence_length + self.num_symbols)

  #   for index in range(input_sequence.shape[1]):
  #     print(f"one_hot_encoding shape: {one_hot_encoding.shape}")
  #     print(f"range(batch_size) shape: {len(range(batch_size))}")
  #     print(f"index: {index}")
  #     print(f"self.max_sequence_length + input_as_numpy[:, index] shape: {(self.max_sequence_length + input_as_numpy[:, index]).shape}")
  #     one_hot_encoding[range(batch_size), index, self.max_sequence_length + input_as_numpy[:, index]] = 1


  #   # Apply the embedding to the input
  #   embedded_input_sequence = Tensor(one_hot_encoding, device=input_sequence.device).dot(self.embedding).reshape(shape=(batch_size, input_sequence.shape[1], -1))
  #   # Add the positional encoding to the embedded input
  #   input_with_position = embedded_input_sequence + self.positional_encoder(embedded_input_sequence.shape[1])
  #   # Pass the input through the transformer blocks
  #   output_sequence = input_with_position.sequential(self.transformer_blocks)
  #   # Apply the final linear layer and softmax
  #   logits = output_sequence.reshape(shape=(-1, output_sequence.shape[-1])).dot(self.final_projection).log_softmax()
  #   # Reshape the output
  #   output_sequence = logits.reshape(shape=(batch_size, -1, logits.shape[-1]))
  #   return output_sequence
