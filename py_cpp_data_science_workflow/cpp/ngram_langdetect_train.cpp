#include <mlpack.hpp>
#include "ngram_langdetect_model.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5)
  {
    cerr << "Usage: " << argv[0] << " data.txt labels.csv model.bin "
        << "[bigramsToKeep]" << endl;
    cerr << " - data.txt should contain individual lines of text" << endl;
    cerr << " - labels.csv should be a one-column CSV with class labels for "
        << "each training point" << endl;
    cerr << " - if not specified, bigramsToKeep is 768" << endl;
    cerr << " - model.bin (or whatever filename) stores the trained model"
        << endl;
    cerr << " - data will be split 80/20 into training/test sets" << endl;
    cerr << " - overall accuracy on test set will be printed" << endl;
    return 1;
  }

  const string dataFile = argv[1];
  const string labelsFile = argv[2];
  const string modelFile = argv[3];
  const size_t bigramsToKeep = (argc == 5) ? atoi(argv[4]) : 768;

  // Load the data.
  fstream f(dataFile.c_str());
  if (!f.is_open())
  {
    cerr << "Could not open '" << dataFile.c_str() << "!" << endl;
    return 1;
  }

  vector<string> data;
  while (!f.eof())
  {
    if (!f.good())
    {
      cerr << "Error reading from '" << dataFile.c_str() << "!" << endl;
      return 1;
    }

    string line;
    data.push_back(line);
    getline(f, data[data.size() - 1]);

    // Delete an empty line.
    if (data[data.size() - 1].size() == 0)
      data.pop_back();
  }
  f.close();

  // Load the labels.
  unordered_map<string, size_t> classMappings;
  Row<size_t> labels(data.size());
  size_t index = 0;

  fstream lf(labelsFile.c_str());
  if (!lf.is_open())
  {
    cerr << "Could not open '" << labelsFile.c_str() << "!" << endl;
    return 1;
  }

  while (!lf.eof())
  {
    if (!lf.good())
    {
      cerr << "Error reading from '" << labelsFile.c_str() << "!" << endl;
      return 1;
    }

    string line;
    getline(lf, line);
    if (line.size() != 0)
    {
      if (classMappings.count(line) == 0)
      {
        const size_t nextIndex = classMappings.size();
        classMappings[line] = nextIndex;
      }

      labels[index] = classMappings[line];
      ++index;
    }
  }

  vector<string> classNames(classMappings.size(), string(""));
  unordered_map<string, size_t>::const_iterator it = classMappings.begin();
  while (it != classMappings.end())
  {
    classNames[(*it).second] = (*it).first;
    ++it;
  }

  // Construct training and test datasets.
  Row<size_t> order = linspace<Row<size_t>>(0, data.size() - 1, data.size());
  Row<size_t> trainOrder, testOrder, trainLabels, testLabels;

  data::Split(order, labels, trainOrder, testOrder, trainLabels, testLabels,
      0.2);

  // Gather training data and test data.
  vector<string> trainData, testData;
  for (size_t i = 0; i < trainOrder.n_elem; ++i)
    trainData.push_back(data[(size_t) trainOrder[i]]);
  for (size_t i = 0; i < testOrder.n_elem; ++i)
    testData.push_back(data[(size_t) testOrder[i]]);

  NGramLangDetectModel m;
  m.Train(trainData, trainLabels, classNames, classNames.size(), bigramsToKeep);

  // Now compute test accuracy.
  Row<size_t> trainPredictions;
  m.Classify(trainData, trainPredictions);
  const double trainAccuracy = 100.0 *
      ((double) accu(trainPredictions == trainLabels)) / trainLabels.n_elem;

  Row<size_t> testPredictions;
  m.Classify(testData, testPredictions);
  const double testAccuracy = 100.0 *
      ((double) accu(testPredictions == testLabels)) / testLabels.n_elem;

  cout << "Training set accuracy: " << trainAccuracy << "\%." << endl;
  cout << "Test set accuracy:     " << testAccuracy << "\%." << endl;

  data::Save(modelFile, "model", m, true /* fatal on failure */);
}
