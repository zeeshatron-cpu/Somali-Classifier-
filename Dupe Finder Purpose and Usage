Step 1.1 – Remove Duplicates from Somali Name Dataset // explaining use of DupeFinder
This script is the first substep in preparing data for training a binary classifier on Somali names using LLaMA 3.1. The goal here is simple: clean up the dataset by removing duplicate names before we move on to labeling and training.
What it does
- Scans a list of Somali names (one per line)
- Finds and removes duplicates
- Optionally normalizes names (e.g. lowercasing, trimming spaces)
- Outputs a cleaned version of the dataset
- Can print a quick summary of how many duplicates were found
Why this matters
Duplicates can mess with model training—especially in a binary classification task—by skewing the data distribution and making the model overfit. This step helps make sure each name is only represented once so the model learns from a balanced, clean dataset.
How to use it 
You'll need Python 3.8+ and pandas installed.
python find_duplicates.py --input somali_names.csv --output cleaned_names.csv


Optional flags:
- --normalize → makes all names lowercase and trims whitespace before checking for duplicates
- --report → prints how many duplicates were found and removed
Input format
A CSV file with one name per line, like:
Ayaan
AYaan
Mohamed
Mohamed

If you use --normalize, both "Ayaan" and "AYaan" will be treated as the same name.
Output
A cleaned CSV file with only unique names. This file is what you'll use for the next step: labeling the data for training.
Next steps
Once you've cleaned the data, move on to Step 1.2: labeling the names for your binary classification task (e.g., gender, origin, etc.)


