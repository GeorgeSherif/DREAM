import pandas as pd

csv_file_path = "./MSA_ALG.csv"
original_data = pd.read_csv(csv_file_path)

csv_file_path = "../../Generated/MSA_and_Algerian.csv"
generated_data = pd.read_csv(csv_file_path)

generated_MSA = generated_data['MSA']
generated_dialect = generated_data['Algerian']
id = "-1"
split = "generated-6-train"
# Create a DataFrame
data = {
    "id": [id] * len(generated_MSA),  # Repeat the id for each entry
    "split": [split] * len(generated_MSA),  # Repeat the split for each entry
    "MSA": generated_MSA,
    "ALG": generated_dialect
}

df_new = pd.DataFrame(data)
df_combined = pd.concat([original_data, df_new], ignore_index=True)

df_combined.to_csv("./MSA_ALG_Augmented.csv")