import pandas as pd
import os

file_path = "./MADAR.corpus.MSA.tsv"
num_prompts = 20
Madar = pd.read_csv(file_path, sep='\t')
Madar = Madar[400:].reset_index(drop=True)
Madar['Done'] = False
Madar.head(20)


prompts = list()
for i in range(num_prompts):
  sentence = Madar['sent'][i]
  prompt_for_finetune = f'''Output only the required text, without any introductory sentences. 
Using the sentence provided in MSA, generate a 4-sentences short story in MSA and then translate it into the following Arabic dialects: 
Jordanian
The format should be as following:

  MSA: generated_story <SEP>
  Jordanian: Jordanian_story
  {sentence}
  '''
  prompts.append(prompt_for_finetune)
print(prompts[0])

#Using the provided sentence in Egyptian dialect as a starting point, create a brief narrative in the same dialect.
#The story should begin with the given sentence and should be composed of a total of five sentences.
#Please ensure that the entire story is written in Egyptian dialect without altering its linguistic style.
# القصة هتبدأ بالجملة دي وهتتكون من أربع جمل كلها باللهجة المصرية، مع الحفاظ على أسلوبها اللغوي.'''


from openai import OpenAI
from tqdm import tqdm
client = OpenAI(api_key="",)
Generated = pd.DataFrame()
Original = pd.DataFrame()
responses = []
for i in tqdm(range(num_prompts)):
  if Madar.at[i, 'Done']  == False:
    prompt = prompts[i]
    # print(prompt)
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        # max_tokens=150
      )
    # print(completion.choices[0].message.content)
    generated_text = completion.choices[0].message.content
    Madar.at[i, 'Done'] = True
    Generated.at[i, 'Prompt'] = prompt
    Generated.at[i, 'Orig_Text'] = Madar.at[i, 'sent']
    MSA = generated_text.split("<SEP>")[0].replace("MSA:","")
    Jordanian = generated_text.split("<SEP>")[1].replace("Jordanian:","")
    # Algerian = generated_text.split("<SEP>")[2].replace("Algerian:","")
    # Syrian = generated_text.split("<SEP>")[3].replace("Syrian:","")
    # Iraqi = generated_text.split("<SEP>")[4].replace("Iraqi:","")
    # Yemeni = generated_text.split("<SEP>")[5].replace("Yemeni:","")
  

    Generated.at[i, 'MSA'] = MSA
    Generated.at[i, 'Jordanian'] = Jordanian
    # Generated.at[i, 'Algerian'] = Algerian
    # Generated.at[i, 'Syrian'] = Syrian
    # Generated.at[i, 'Iraqi'] = Iraqi
    # Generated.at[i, 'Yemeni'] = Yemeni

    responses.append(generated_text)



Madar.to_csv(file_path, sep='\t', index=False, header=True)
generated_path = "./Generated_2nd.xlsx"

# Check if the file exists and is not empty
if os.path.exists(generated_path) and os.path.getsize(generated_path) > 0:
    # Read existing data
    existing_data = pd.read_excel(generated_path)
    # Concatenate new data with existing data
    combined_data = pd.concat([existing_data, Generated], ignore_index=True)
else:
    # If the file does not exist or is empty, use the new data
    combined_data = Generated

# Save the combined data back to the Excel file
combined_data.to_excel(generated_path, index=False)