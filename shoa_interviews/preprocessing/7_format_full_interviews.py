import os
import re

from tqdm import tqdm

workdir = r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews'
savedir = r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews\new'
for fname in tqdm(os.listdir(workdir)):
    if fname.endswith('txt'):
        with open(workdir + os.sep + fname, 'r', encoding='utf-8') as f:
            content = f.read()
            content = re.sub(r'([A-Z][A-Z ]+: )', r'\n\1', content)[1:]

        with open(savedir + os.sep + fname, 'w', encoding='utf-8') as f:
            f.write(content)
