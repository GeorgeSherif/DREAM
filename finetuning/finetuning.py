#This code finetunes the model, and create the generation csv file from MSA to dialect
print("Started Importing")
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import argparse
nltk.download('punkt')
from tqdm import tqdm
import evaluate


parser = argparse.ArgumentParser(description='AraBERT')
parser.add_argument('model', type=str)
parser.add_argument('path_to_csv', type=str)

# Parse arguments
args = parser.parse_args()
model_name = args.model

print("Loading the model")
# Load the model and tokenizer
checkpoint = model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

# Load the data
csv_file_path = args.path_to_csv
df = pd.read_csv(csv_file_path)
df = df[~df['split'].str.contains("corpus-6-test")]
print("Data loaded, total records:", len(df))

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")

# Convert DataFrames to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Function to preprocess the data
def preprocess_function(examples):
    inputs = [ex for ex in examples["MSA"]]
    targets = [ex for ex in examples[df.columns[3]]]  # Assuming the dialect column is in the same position
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the function to preprocess the data
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",              # output directory
    evaluation_strategy="epoch",         # evaluate each epoch
    learning_rate=2e-5,                  # learning rate
    per_device_train_batch_size=16,      # batch size for training
    per_device_eval_batch_size=16,       # batch size for evaluation
    num_train_epochs=3,                  # number of training epochs
    weight_decay=0.01,                   # strength of weight decay
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()

print("Training completed")

# # Optionally, save the model
# model.save_pretrained("./finetuned_model")
# tokenizer.save_pretrained("./finetuned_model")
# torch.save({
#     "optimizer": trainer.optimizer.state_dict(),
#     "scheduler": trainer.lr_scheduler.state_dict(),
# }, "./finetuned_model/optimizer_scheduler.pt")
# print("Model saved")

df = pd.read_csv(csv_file_path)
print(len(df))
df = df[df['split'].str.contains("corpus-6-test")]
print(len(df))

# Prepare the model for CUDA execution
if torch.cuda.is_available():
    model.cuda()

print("Started translation")
results = []

dialect_target = df.columns[4]
print(dialect_target)

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
save_name = checkpoint.split("/")[1]
results_df.to_csv(f'./finetuned_AUG_inference_{dialect_target}_{save_name}.csv', index=False)

print(results_df.head())

df = results_df

# Data provided
reference_texts = df[dialect_target]
translated_texts = df[f"{dialect_target} Translation"]

# Tokenizing the texts
references = [word_tokenize(ref) for ref in reference_texts]
hypotheses = [word_tokenize(hyp) for hyp in translated_texts]

# Calculating BLEU scores for each pair
bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(references, hypotheses)]

# Calculating average BLEU score
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu_score * 100:.2f}%")

# Load metrics
chrf = evaluate.load("chrf")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


# Data provided
reference_texts = df[dialect_target]
translated_texts = df[f"{dialect_target} Translation"]

# Tokenizing the texts might not be necessary for all metrics, but doing so for consistency
references = [word_tokenize(ref) for ref in reference_texts]
hypotheses = [word_tokenize(hyp) for hyp in translated_texts]

# Calculate metrics
results = {
    "chrf": [],
    "sacrebleu": [],
    "rouge": [],
    "meteor": []
}

for ref, hyp in zip(references, hypotheses):
    ref_str = " ".join(ref)  # Convert list of tokens back to string
    hyp_str = " ".join(hyp)
    results["chrf"].append(chrf.compute(predictions=[hyp_str], references=[ref_str])['score'])
    results["sacrebleu"].append(sacrebleu.compute(predictions=[hyp_str], references=[ref_str])['score'])
    results["rouge"].append(rouge.compute(predictions=[hyp_str], references=[ref_str]))
    results["meteor"].append(meteor.compute(predictions=[hyp_str], references=[ref_str])['meteor'])

# Display or process the results as needed
for metric, scores in results.items():
    if metric != "rouge":
        average_score = sum(scores) / len(scores)
        print(f"Average {metric}: {average_score:.4f}")
    else:
        # ROUGE outputs multiple scores, handle differently
        rouge_scores = {key: sum(d[key] for d in scores) / len(scores) for key in scores[0]}
        print(f"Average ROUGE scores: {rouge_scores}")
