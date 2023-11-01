import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLangDetectModel(nn.Module):
  def __init__(self):
    super(NGramLangDetectModel, self).__init__()
    self.model = None
    self.bigram_dimension_map = {}
    self.class_names = []

  def _compute_ngrams_initial(self, string_data, bigrams_to_keep):
    # We will collect all single-character counts, and keep all of those.
    # We will collect the top 768 most common 2-grams.
    # We really care about just byte values, no locale handling.

    bigram_dimension_map = {}

    # First, collect all of the bigrams we have, so that we can select the 768
    # most common ones.
    bigram_map = {}
    for string in string_data:
      char_array = bytearray(string, 'utf-8')
      for j in range(len(char_array) - 1):
        key = (char_array[j] << 8) + char_array[j + 1]
        bigram_map[key] = bigram_map.get(key, 0) + 1

    # Now sort and find the 768 most common bigrams.
    bigram_counts = np.array(list(bigram_map.items()))
    ordering = np.argsort(-bigram_counts[:, 1])  # Sort in descending order.
    kept_bigrams = min(bigrams_to_keep, len(ordering))

    # Map bigrams to their indices in the matrix (if the bigram exists).
    for i in range(kept_bigrams):
      key = bigram_counts[ordering[i], 0]
      bigram_dimension_map[key] = i

    return (bigram_dimension_map,
            self._compute_ngrams(string_data, bigram_dimension_map))

  def _compute_ngrams(self, string_data, bigram_dimension_map):
    num_features = 256 + len(bigram_dimension_map)
    num_samples = len(string_data)

    # Resize the matrix to the right size.
    ngrams = np.zeros((num_samples, num_features))

    for i in range(num_samples):
      char_array = bytearray(string_data[i], 'utf-8')
      for j in range(len(char_array) - 1):
        bigram_key = (char_array[j] << 8) + char_array[j + 1]

        # Check if the bigram exists in the dimension map.
        if bigram_key in bigram_dimension_map:
          ngrams[i, 256 + bigram_dimension_map[bigram_key]] += 1

        ngrams[i, char_array[j]] += 1

      # Don't forget the last character!
      ngrams[i, char_array[-1]] += 1

      # Now normalize.
      ngrams[i, :] /= np.sum(ngrams[i, :])

    return ngrams

  def forward(self, data):
    result = F.log_softmax(self.linear(data), dim=-1)
    return result

  def fit(self, data, labels, class_names, num_classes, bigrams_to_keep=768):
    self.class_names = class_names

    # Prepare the data for training.
    (self.bigram_dimension_map, ngram_data) = self._compute_ngrams_initial(data,
        bigrams_to_keep)
    ngram_data = torch.Tensor(ngram_data)

    self.linear = nn.Linear(ngram_data.shape[1], len(class_names))

    # Train a logistic regression model.
    optimizer = optim.SGD(self.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(25):
      print("Epoch", epoch)
      for instance, label in zip(ngram_data, labels):
        optimizer.zero_grad()
        y_pred = self.forward(instance)
        label_vec = torch.zeros(y_pred.shape[0])
        label_vec[label] = 1.0
        loss = criterion(y_pred, label_vec)
        loss.backward()
        optimizer.step()

  def predict(self, data):
    ngram_data = self._compute_ngrams(data, self.bigram_dimension_map)
    ngram_data = torch.Tensor(ngram_data)
    return np.array(torch.argmax(self.forward(ngram_data), dim=-1))

  def class_name(self, i):
    if i < len(self.class_names):
      return self.class_names[i]
    else:
      return "unknown"
