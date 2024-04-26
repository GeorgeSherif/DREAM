print("Started Importing")
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
print("Finished Importing")

parser = argparse.ArgumentParser(description='AraBERT')
parser.add_argument('path_to_data', type=str, help='Path to generated sentences.')
parser.add_argument('soft_labeling', type=lambda x: (str(x).lower() == 'true'), help='Whether we will use soft labeling or Mean reciprocal rank')
parser.add_argument('--number_labels', type=int, default=2, help='Number of soft labels if we are using soft labeling')

# Parse arguments
args = parser.parse_args()

# Use the arguments in your program
print(f"Path to data: {args.path_to_data}")
print(f"Soft labeling: {args.soft_labeling}")
print(f"Number of labels: {args.number_labels}")

args = parser.parse_args()

path_to_data = args.path_to_data
generated_test = pd.read_csv(path_to_data)

file_pattern = './Data/*.tsv'

file_paths = glob.glob(file_pattern)
# Read each file into a DataFrame and store them in a list
dfs = [pd.read_csv(file_path, sep='\t', nrows=2000) for file_path in file_paths]
# dfs = [pd.read_csv(file_path, sep='\t') for file_path in file_paths]

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop(['sentID.BTEC','split'], axis=1, inplace=True)

label_dict = {label: idx for idx, label in enumerate(merged_df['lang'].unique())}
print(label_dict)

merged_df['lang'] = merged_df['lang'].map(label_dict)


generated_test['lang'] = generated_test['lang'].replace({'Egyptian': 'EGY', 'Moroccan': 'MOR', 'Saudi': 'SAU', 'Lebanese': 'LEB', "Algerian": "ALG", "Syrian": "SYR", "Sudanese":"SUD", "Iraqi": "IRQ", "Yemeni":"YEM"})
generated_test['lang'] = generated_test['lang'].map(label_dict)
generated_test = generated_test.sample(frac=1, random_state=42).reset_index(drop=True)

df_train, temp = train_test_split(merged_df, test_size=0.1, random_state=42, stratify=merged_df['lang'])
df_val, df_test = train_test_split(temp, test_size=0.8, random_state=42, stratify=temp['lang'])

# Initialize tokenizer
model_name = "aubmindlab/bert-base-arabert"
#"aubmindlab/bert-base-arabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize and include labels
def tokenize_and_label_function(examples):
    tokenized_inputs = tokenizer(examples['sent'], truncation=True, padding='max_length', max_length=512)
    tokenized_inputs['labels'] = examples['lang']
    return tokenized_inputs

# Prepare datasets
train_dataset = Dataset.from_pandas(df_train).map(tokenize_and_label_function, batched=True)
val_dataset = Dataset.from_pandas(df_val).map(tokenize_and_label_function, batched=True)
test_dataset = Dataset.from_pandas(df_test).map(tokenize_and_label_function, batched=True)

generated_dataset = Dataset.from_pandas(generated_test).map(tokenize_and_label_function, batched=True)

# Load model with correct number of labels
num_labels = df_train['lang'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=320,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


if args.soft_labeling == True:
    # Compute metrics function
    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_recall_fscore_support(labels, predictions, average='weighted')[0],
            "recall": precision_recall_fscore_support(labels, predictions, average='weighted')[1],
            "f1": precision_recall_fscore_support(labels, predictions, average='weighted')[2],
        }
else: 
    def mean_reciprocal_rank(predictions, labels):
        """Calculate Mean Reciprocal Rank (MRR)"""
        ranks = []
        for i, label in enumerate(labels):
            rank = np.where(predictions[i] == label)[0][0] + 1
            ranks.append(1.0 / rank)
        return np.mean(ranks)

    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        softmax_scores = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
        softmax_scores_np = softmax_scores.numpy()
        sorted_indices = np.argsort(softmax_scores_np, axis=1)[:, ::-1]  # Descending order
        mrr = mean_reciprocal_rank(sorted_indices, labels)
        return {
            "accuracy": accuracy_score(labels, np.argmax(predictions, axis=1)),
            "precision": precision_recall_fscore_support(labels, np.argmax(predictions, axis=1), average='weighted')[0],
            "recall": precision_recall_fscore_support(labels, np.argmax(predictions, axis=1), average='weighted')[1],
            "f1": precision_recall_fscore_support(labels, np.argmax(predictions, axis=1), average='weighted')[2],
            "mrr": mrr
        }


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Started training")
trainer.train()
test_results = trainer.evaluate(test_dataset)

print("Test Results:", test_results)



if args.soft_labeling:

    predictions = trainer.predict(test_dataset)
    softmax_scores = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
    softmax_scores_np = softmax_scores.numpy()
    top2_preds = np.argsort(softmax_scores_np, axis=1)[:, -args.number_labels:]
    true_labels = predictions.label_ids
    correct_top2 = [true_labels[i] in top2_preds[i] for i in range(len(true_labels))]
    top2_accuracy = np.mean(correct_top2)

    print(f"Top-{args.number_labels} Accuracy on the Test Dataset: {top2_accuracy}")


    predictions = trainer.predict(generated_dataset)
    softmax_scores = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
    softmax_scores_np = softmax_scores.numpy()
    top2_preds = np.argsort(softmax_scores_np, axis=1)[:, -args.number_labels:]
    true_labels = predictions.label_ids
    correct_top2 = [true_labels[i] in top2_preds[i] for i in range(len(true_labels))]
    top2_accuracy = np.mean(correct_top2)

    print(f"Top-{args.number_labels} Accuracy on the Generated Dataset: {top2_accuracy}")

else:
    predictions = trainer.predict(test_dataset)
    print(f"MRR on the Test Dataset: {predictions.metrics}")

    predictions = trainer.predict(generated_dataset)
    print(f"MRR on the Generated Dataset: {predictions.metrics}")


# trainer.push_to_hub("AraBERT-MADAR")
