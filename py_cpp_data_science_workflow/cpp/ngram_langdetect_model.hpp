#ifndef NGRAM_LANGDETECT_MODEL_HPP
#define NGRAM_LANGDETECT_MODEL_HPP

#include <mlpack.hpp>
#include "ngram.hpp"

class NGramLangDetectModel
{
 public:
  void Train(const std::vector<std::string>& data,
             const arma::Row<size_t>& labels,
             const std::vector<std::string>& classNames,
             const size_t numClasses,
             const size_t bigramsToKeep = 768);

  size_t Classify(const std::string& point) const;

  void Classify(const std::string& point,
                size_t& prediction,
                arma::rowvec& probabilities) const;

  void Classify(const std::vector<std::string>& data,
                arma::Row<size_t>& predictions) const;

  void Classify(const std::vector<std::string>& data,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities) const;

  // Maybe a little awkward to put this here...
  std::string ClassName(const size_t i) const;

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  mlpack::SoftmaxRegression model;
  std::unordered_map<uint16_t, size_t> bigramDimensionMap;
  std::vector<std::string> classNames;
};

// Include implementation.
#include "ngram_langdetect_model_impl.hpp"

#endif
