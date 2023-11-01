CXX := g++
CXXFLAGS += -std=c++17 -O3 -DNDEBUG -DARMA_DONT_USE_WRAPPER -fopenmp
#CXXFLAGS += -std=c++17 -O0 -g -DDEBUG -DARMA_DONT_USE_WRAPPER -fopenmp
CXXFLAGS += -I/home/ryan/src/mlpack/src/ -I/home/ryan/src/ensmallen/include/

LDFLAGS += -fopenmp
LDFLAGS += -lopenblas

SOURCES := ngram.hpp ngram_impl.hpp ngram_langdetect_model.hpp ngram_langdetect_model_impl.hpp

default: all

all: ngram_langdetect_train ngram_langdetect_run

ngram_langdetect_train: ngram_langdetect_train.cpp $(SOURCES)
	$(CXX) $(CXXFLAGS) ngram_langdetect_train.cpp -o ngram_langdetect_train $(LDFLAGS) $(LIBS)

ngram_langdetect_run: ngram_langdetect_run.cpp $(SOURCES)
	$(CXX) $(CXXFLAGS) ngram_langdetect_run.cpp -o ngram_langdetect_run $(LDFLAGS) $(LIBS)

clean:
	rm -f ngram_langdetect_train ngram_langdetect_run
