print("Started Importing")
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import argparse
from tqdm import tqdm
print("Done Importing")

parser = argparse.ArgumentParser(description='Merge two data files on sentence ID.')
parser.add_argument('data', help='Path to the MSA to Dialect file.')
parser.add_argument('model')

args = parser.parse_args()

# Define a dictionary mapping dialect abbreviations to their full names
dialects = {
    'EGY': 'Egyptian Arabic'
}

print("Loading the model")
checkpoint = args.model
save_name = checkpoint.split("/")[1]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

# Load the data
csv_file_path = args.data
df = pd.read_csv(csv_file_path)
print(len(df))
df = df[df['split'].str.contains("corpus-6-test")]
print(len(df))

# Prepare the model for CUDA execution
if torch.cuda.is_available():
    model.cuda()

print("Started translation")
results = []

dialect_target = df.columns[3]

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Translating"):
    msa_sentence = row['MSA']
    dialect_sentence = row[dialect_target]
    prompt = f"{msa_sentence}\nTranslate this sentence from Modern Standard Arabic to {dialect_target}"

    # Encoding the input text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    # Generating output sequences
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store results
    results.append({
        'id': row['id'],
        'split': row['split'],
        'MSA': msa_sentence,
        f'{dialect_target}': dialect_sentence,
        f'{dialect_target} Translation': translated_text
    })

# Optionally, convert results to DataFrame and save or print
results_df = pd.DataFrame(results)
print(results_df.head())
results_df.to_csv(f'./translated_zero_shot/inference_{dialect_target}_{save_name}.csv', index=False)
