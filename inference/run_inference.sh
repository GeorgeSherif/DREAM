python prepare_files.py ../Unmodified_Data/MADAR.corpus.Cairo.tsv ../Data/MADAR.corpus.MSA.tsv

python inference.py ./formatted_files/MSA_EGY.csv bigscience/mt0-base

python evaluation.py ./translated_zero_shot/inference_EGY_mt0-base.csv EGY
