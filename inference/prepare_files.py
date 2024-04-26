import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Merge two data files on sentence ID.')
parser.add_argument('dialect_file', help='Path to the dialect file.')
parser.add_argument('msa_file', help='Path to the MSA file.')
args = parser.parse_args()


# Read the files into DataFrames
df_egy = pd.read_csv(args.dialect_file, sep='\t', header=None, names=['sentID.BTEC', 'split', 'lang', 'sent'])
df_msa = pd.read_csv(args.msa_file, sep='\t', header=None, names=['sentID.BTEC', 'split', 'lang', 'sent'])

# Merge the DataFrames on 'sentID.BTEC' and 'split' (assuming split is always the same for matching IDs) and exclude non-matching entries
merged_df = pd.merge(df_egy, df_msa, on=['sentID.BTEC', 'split'], how='inner')
merged_df = merged_df.drop(merged_df.index[0])

language_X = merged_df['lang_x'][5]
language_Y = merged_df['lang_y'][5]
ids = merged_df['sentID.BTEC']
sent_X = merged_df['sent_x']
sent_Y = merged_df['sent_y']

final_df = pd.DataFrame({
    'id': merged_df['sentID.BTEC'],
    'split': merged_df['split'],
    f'{language_Y}': sent_Y,
    f'{language_X}': sent_X,
})

final_df.to_csv(f'./formatted_files/MSA_{language_X}.csv', index=False)
print("Done")