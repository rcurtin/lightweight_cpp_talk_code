import pickle
import sys
from NGramLangDetect import NGramLangDetectModel

if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "model.pkl")
  print(" - once loaded, the program reads from stdin")
  print(" - when a newline is encountered, the predicted language is " +
      "printed")
  print(" - ctrl+c or ctrl+d to exit")
  sys.exit(1)

model_file = sys.argv[1]
model = pickle.load(open(model_file, 'rb'))

while True:
  try:
    line = input()
    class_prediction = model.predict(line)
    print(model.class_name(class_prediction[0]))
  except EOFError:
    break
