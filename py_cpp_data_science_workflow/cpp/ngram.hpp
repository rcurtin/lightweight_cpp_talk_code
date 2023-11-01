#ifndef NGRAM_HPP
#define NGRAM_HPP

#include <mlpack.hpp>

template<typename MatType>
void ComputeNGrams(const std::vector<std::string>& stringData,
                   std::unordered_map<uint16_t, size_t>& bigramDimensionMap,
                   MatType& ngrams,
                   const size_t bigramsToKeep = 768);

template<typename MatType>
void ComputeNGrams(const std::vector<std::string>& stringData,
                   MatType& ngrams,
                   const std::unordered_map<uint16_t, size_t> bigramDimensionMap);

template<typename MatType>
void ComputeNGrams(const std::string& stringData,
                   MatType& ngrams,
                   const std::unordered_map<uint16_t, size_t> bigramDimensionMap);

#include "ngram_impl.hpp"

#endif
