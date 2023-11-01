import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from NGramLangDetect import NGramLangDetectModel

if len(sys.argv) != 4 and len(sys.argv) != 5:
  print("Usage:", sys.argv[0], "data.txt labels.csv model.pkl " +
      "[bigramsToKeep]")
  print(" - data.txt should contain individual lines of text")
  print(" - labels.csv should be a one-column CSV with class labels for " +
      "each training point")
  print(" - if not specified, bigramsToKeep is 768")
  print(" - model.pkl (or whatever filename) stores the trained model")
  print(" - data will be split 80/20 into training/test sets")
  print(" - overall accuracy on test set will be printed")
  sys.exit(1)

data_file = sys.argv[1]
labels_file = sys.argv[2]
model_file = sys.argv[3]
bigrams_to_keep = int(sys.argv[4]) if len(sys.argv) == 5 else 768

# Load the data.
with open(data_file, 'r') as f:
  data = [line for line in f if line.strip()]

# Load the labels.
class_mappings = {}
labels = []
index = 0

with open(labels_file, 'r') as lf:
  for line in lf:
    line = line.strip()
    if len(line) != 0:
      if line not in class_mappings:
        next_index = len(class_mappings)
        class_mappings[line] = next_index
      labels.append(class_mappings[line])
      index += 1

class_names = [None] * len(class_mappings)
for key, value in class_mappings.items():
  class_names[value] = key

# Construct training and test datasets.
train_data, test_data, train_labels, test_labels = train_test_split(data,
    labels, test_size=0.2)

model = NGramLangDetectModel.NGramLangDetectModel()
model.fit(train_data, train_labels, class_names, len(class_mappings),
    bigrams_to_keep)

# Now compute test accuracy.
train_predictions = model.predict(train_data)
train_accuracy = 100.0 * np.sum(train_predictions == train_labels) / \
    len(train_labels)

test_predictions = model.predict(test_data)
test_accuracy = 100.0 * np.sum(test_predictions == test_labels) / \
    len(test_labels)

print("Training set accuracy: {:.2f}%".format(train_accuracy))
print("Test set accuracy: {:.2f}%".format(test_accuracy))

pickle.dump(model, open(model_file, 'wb'))
