from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Merge two data files on sentence ID.')
parser.add_argument('dialect_file', help='Path to the dialect file.')
parser.add_argument('dialect')

args = parser.parse_args()

csv_file_path = args.dialect_file
df = pd.read_csv(csv_file_path)


# Data provided
reference_texts = df[f"{args.dialect}"]
translated_texts = df[f"{args.dialect} Translation"]

# Tokenizing the texts
references = [word_tokenize(ref) for ref in reference_texts]
hypotheses = [word_tokenize(hyp) for hyp in translated_texts]

# Calculating BLEU scores for each pair
bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(references, hypotheses)]

# Calculating average BLEU score
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(average_bleu_score, bleu_scores)