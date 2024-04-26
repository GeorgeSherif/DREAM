# DREAM 💭: Dialect Rewriting Enhancement for Arabic-pretrained Models
This project aims to improve the capabilities of Large Language Models for rewriting MSA text to Arabic dialects. In this project, we use MADAR corpus as the backbone dataset, while augmenting it with extra generated data using ChatGPT 4.

## How to run the code
### Generating new samples
Add your openAI key in the Data_Augmentation.py and modify the prompt to the desired dialects, and simply run it using python Data_Augmentation.py.

### Finetuning AraBERT
Adjust AraBERT.sh with the desired settings as follows:
  1) The first attribute is the path to the generated data from the previous step.
  2) The second attribute is a boolean to state whether the evaluation will be top-N (True) or Mean Reciprocal Rank (False).
  3) If the second attribute is True, then identify the N for the top-N evaluation in the third attribute.
Simply run bash `AraBERT.sh`.

### Running inference on mt0-small and mt0-base
cd inference
adjust the
