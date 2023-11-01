FROM pytorch/pytorch

RUN mkdir NGramLangDetect
ADD NGramLangDetect/NGramLangDetectModel.py NGramLangDetect/
ADD NGramLangDetect/__init__.py NGramLangDetect/
ADD all_lang_data_model.pkl .
ADD ngram_lang_detect_run.py .

ENTRYPOINT python3 ngram_lang_detect_run.py all_lang_data_model.pkl
