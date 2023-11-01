#ifndef NGRAM_IMPL_HPP
#define NGRAM_IMPL_HPP

// In case it hasn't been included yet.
#include "ngram.hpp"

#include <locale>

template<typename MatType>
void ComputeNGrams(const std::vector<std::string>& stringData,
                   std::unordered_map<uint16_t, size_t>& bigramDimensionMap,
                   MatType& ngrams,
                   const size_t bigramsToKeep)
{
  // we will collect all single-character counts, and keep all of those
  // we will collect the top 768 most common 2-grams
  // we really care about just byte values, no locale handling

  // First, collect all of the bigrams we have, so that we can select the 768
  // most common ones.
  std::unordered_map<uint16_t, size_t> bigramMap;
  for (size_t i = 0; i < stringData.size(); ++i)
  {
    for (size_t j = 0; j < stringData[i].size() - 1; ++j)
    {
      uint16_t key = ((unsigned char) stringData[i][j] << 8) +
                     ((unsigned char) stringData[i][j + 1]);
      ++bigramMap[key];
    }
  }

  // Now sort and find the 768 most common bigrams.
  arma::umat bigramCounts(bigramMap.size(), 2); // Row-major for faster sorting.
  std::unordered_map<uint16_t, size_t>::const_iterator it = bigramMap.begin();
  size_t index = 0;
  while (it != bigramMap.end())
  {
    bigramCounts(index, 0) = (*it).first;
    bigramCounts(index, 1) = (*it).second;

    ++index;
    ++it;
  }

  arma::uvec ordering = arma::stable_sort_index(bigramCounts.col(1), "descend");
  const size_t keptBigrams = std::min(bigramsToKeep, (size_t) ordering.n_elem);

  // Map bigrams to their indices in the matrix (if the bigram exists).
  bigramDimensionMap.clear();
  for (size_t i = 0; i < keptBigrams; ++i)
  {
    const uint16_t key = (uint16_t) bigramCounts(ordering[i], 0);
    bigramDimensionMap[key] = i;
  }

  ComputeNGrams(stringData, ngrams, bigramDimensionMap);
}

template<typename MatType>
void ComputeNGrams(const std::vector<std::string>& stringData,
                   MatType& ngrams,
                   const std::unordered_map<uint16_t, size_t> bigramDimensionMap)
{
  // Resize the matrix to the right size.
  // This assumes the first 256 dimensions will be for single-byte features.
  ngrams.zeros(256 + bigramDimensionMap.size(), stringData.size());
  for (size_t i = 0; i < stringData.size(); ++i)
  {
    for (size_t j = 0; j < stringData[i].size() - 1; ++j)
    {
      uint16_t bigramKey = ((unsigned char) stringData[i][j] << 8) +
                           ((unsigned char) stringData[i][j + 1]);
      if (bigramDimensionMap.count(bigramKey) > 0)
      {
        ++ngrams(256 + bigramDimensionMap.at(bigramKey), i);
      }

      ++ngrams((unsigned char) stringData[i][j], i);
    }

    // Don't forget the last character!
    ++ngrams((unsigned char) stringData[i][stringData[i].size() - 1], i);

    // Now normalize.
    ngrams.col(i) /= arma::accu(ngrams.col(i));
  }
}

template<typename MatType>
void ComputeNGrams(const std::string& stringData,
                   MatType& ngrams,
                   const std::unordered_map<uint16_t, size_t> bigramDimensionMap)
{
  // Resize the matrix to the right size.
  // This assumes the first 256 dimensions will be for single-byte features.
  ngrams.zeros(256 + bigramDimensionMap.size(), 1);
  for (size_t j = 0; j < stringData.size() - 1; ++j)
  {
    uint16_t bigramKey = ((unsigned char) stringData[j] << 8) +
                         ((unsigned char) stringData[j + 1]);
    if (bigramDimensionMap.count(bigramKey) > 0)
    {
      ++ngrams(256 + bigramDimensionMap.at(bigramKey), 0);
    }

    ++ngrams((unsigned char) stringData[j], 0);
  }

  // Don't forget the last character!
  ++ngrams((unsigned char) stringData[stringData.size() - 1], 0);

  // Now normalize.
  ngrams.col(0) /= arma::accu(ngrams.col(0));
}

#endif
