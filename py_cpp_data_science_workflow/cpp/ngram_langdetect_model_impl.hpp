#ifndef NGRAM_LANGDETECT_MODEL_IMPL_HPP
#define NGRAM_LANGDETECT_MODEL_IMPL_HPP

#include <mlpack.hpp>
#include "ngram.hpp"

inline void NGramLangDetectModel::Train(
    const std::vector<std::string>& data,
    const arma::Row<size_t>& labels,
    const std::vector<std::string>& classNames,
    const size_t numClasses,
    const size_t bigramsToKeep)
{
  this->classNames = classNames;

  arma::mat trainData;
  ComputeNGrams(data, bigramDimensionMap, trainData, bigramsToKeep);

  // Bug: SoftmaxRegression doesn't set numClasses in Train().
  model = mlpack::SoftmaxRegression(trainData, labels, numClasses, 0.00001,
      true);
}

inline size_t NGramLangDetectModel::Classify(const std::string& point) const
{
  arma::vec processedPoint;
  ComputeNGrams(point, processedPoint, bigramDimensionMap);
  return model.Classify(processedPoint);
}

inline void NGramLangDetectModel::Classify(const std::string& point,
                                           size_t& prediction,
                                           arma::rowvec& probabilities) const
{
  arma::vec processedPoint;
  arma::Row<size_t> predictions(1);

  ComputeNGrams(point, processedPoint, bigramDimensionMap);
  model.Classify(processedPoint, predictions, probabilities);
  prediction = predictions[0];
}

inline void NGramLangDetectModel::Classify(
    const std::vector<std::string>& data,
    arma::Row<size_t>& predictions) const
{
  arma::mat points;
  ComputeNGrams(data, points, bigramDimensionMap);
  model.Classify(points, predictions);
}

inline void NGramLangDetectModel::Classify(
    const std::vector<std::string>& data,
    arma::Row<size_t>& predictions,
    arma::mat& probabilities) const
{
  arma::mat points;
  ComputeNGrams(data, points, bigramDimensionMap);
  model.Classify(points, predictions, probabilities);
}

inline std::string NGramLangDetectModel::ClassName(const size_t i) const
{
  return classNames[i];
}

template<typename Archive>
void NGramLangDetectModel::serialize(Archive& ar,
                                     const unsigned int /* version */)
{
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(bigramDimensionMap));
  ar(CEREAL_NVP(classNames));
}

// Include implementation.
#include "ngram_langdetect_model_impl.hpp"

#endif
