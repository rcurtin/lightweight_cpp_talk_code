#include <mlpack.hpp>
#include "ngram_langdetect_model.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " model_file.bin" << endl;
    cerr << " - once loaded, the program reads from stdin" << endl;
    cerr << " - when a newline is encountered, the predicted language is "
        << "printed" << endl;
    cerr << " - ctrl+c or ctrl+d to exit" << endl;
    return 1;
  }

  const std::string modelFile = argv[1];

  NGramLangDetectModel m;
  data::Load(modelFile, "model", m, true /* fatal */);

  while(true)
  {
    string line;
    getline(cin, line);

    if (cin.eof())
    {
      return 0;
    }

    size_t classPrediction = m.Classify(line);
    cout << m.ClassName(classPrediction) << endl;
  }
}
