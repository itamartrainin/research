import os
import re
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from shoa_interviews.preprocessing import config

#%%
working_dir = r'C:\Data\shoa_dataset\Martha_transcripts\outputs'
save_dir = working_dir

assert os.path.isdir(working_dir)
assert os.path.isdir(save_dir)

#%%
transcripts = pd.read_pickle(save_dir + os.sep + 'contents.pkl')

#%%
full_interviews = transcripts.groupby('interview_id')
# full_interviews = transcripts.groupby('interview_id')['content'].apply(lambda x: ' '.join(x))

#%%
def get_sep(row):
    prev_interview_id, prev_section_id, prev_content_id = row['prev_info']
    interview_id, section_id, content_id = row['interview_id'], row['section_id'], row['content_id']

    if interview_id != prev_interview_id:
        return '<new_doc>'
    elif section_id != prev_section_id:
        return '\n'
    else:
        return ' '

for name, group in full_interviews:
    group['prev_info'] = group[['interview_id', 'section_id', 'content_id']].shift(1).progress_apply(lambda x: (x['interview_id'], x['section_id'], x['content_id']), axis=1)

    group['sep_tkn'] = group[['interview_id', 'section_id', 'content_id', 'prev_info']].progress_apply(
        get_sep, axis=1)

    content = ''
    for i, row in tqdm(group.iterrows(), total=len(transcripts)):
        if row['sep_tkn'] != '<new_doc>':
            content += row['sep_tkn']
        content += row['content']

    with open(r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews' + os.sep + str(name) + '.txt', 'w', encoding='utf-8') as f:
        f.write(content)

