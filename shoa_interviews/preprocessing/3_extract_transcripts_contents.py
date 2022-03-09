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
TRANSCRIPT_XML_TAG = 'transcription'
SECTION_XML_TAG = 'p'

SECTION_REG = re.compile(rf'<{SECTION_XML_TAG}>(.*?)</{SECTION_XML_TAG}>')
CONTENT_REG = re.compile(r"<span m=\'([0-9]+)\'>([^<]+)</span>")

#%%
def get_transcription_content(xml):
    return xml[len(f'<{TRANSCRIPT_XML_TAG}>'):-len(f'</{TRANSCRIPT_XML_TAG}>')]

def get_sections(xml):
    """
    Extact sections (<p>...</p>) from transcript xml
    :param xml: transcript xml where '<transcription>' and '</transcription>' have been removed
    :return: List sections
    """
    # Check prefix and suffix removed
    assert xml[:len(f'<{TRANSCRIPT_XML_TAG}>')] != f'<{TRANSCRIPT_XML_TAG}>'
    assert xml[-len(f'</{TRANSCRIPT_XML_TAG}>'):] != f'</{TRANSCRIPT_XML_TAG}>'

    return SECTION_REG.findall(xml)

def get_section_contents(section):
    """
    Extract the inner data from the xml
    Assumes the format: <span m='[time_milisec]'>[text]</span>
    :param section: A section, that is the part between <p> and </p>
    :return: List of pairs ([time_milisec], [text])
    """
    # Validate that there is no error, that is check is there is no nested <p>
    assert '<p>' not in section
    assert '</p>' not in section

    return CONTENT_REG.findall(section)

#%%
transcripts_files = pd.read_pickle(working_dir + os.sep + 'transcripts.pkl')

#%%
# Extract Transcript content, that is what is whithing the <transcript>...</transcript>
print('Extracting transcripts contents')
transcripts = transcripts_files
transcripts['transcription'] = transcripts[config.TRANS_XML].progress_apply(get_transcription_content)
# WARN
# Clean empty transcripts
transcripts = transcripts[transcripts['transcription'] != ''].reset_index(drop=True)

#%%
# From the transcript, extract the sections (<p>...</p>)
print('Extracting sections')
transcripts['sections'] = transcripts['transcription'].progress_apply(get_sections)
# Convert into a list of sections
transcripts = transcripts.explode('sections')

#@TODO !!! WARN !!! -- maybe should keep empty sections?
# Clean empty sections
transcripts = transcripts[transcripts['sections'] != '']
# Note the transcript's ID
transcripts = transcripts.rename_axis(config.SECTION_ID).reset_index()

#%%
# From each sections extract the contents of the section
print('Extract content')
transcripts['contents'] = transcripts['sections'].progress_apply(get_section_contents)
# Convert into a list of contents
transcripts = transcripts.explode('contents')
# Split contents into columns (Time col, Content col)
pairs = [[int(x[0]), x[1]] for x in transcripts['contents'].tolist()]
transcripts[[config.TIME, config.CONTENT]] = pd.DataFrame(pairs, index=transcripts.index)
# Note the section ID
transcripts = transcripts.rename_axis(config.CONTENT_ID).reset_index()

#%%
# Delete temp columns
print('Deleting Columns')
transcripts = transcripts.drop(columns=[config.TRANS_XML, 'transcription', 'sections', 'contents'])

#%%
# Handle cases where there are multiple words (separated by ' ')
print('Split words')
def separate_jumped_tkns(tkn):
    tkn = tkn.strip()
    if (len(tkn) == 0) or ('[' in tkn) or (']' in tkn):
        # Special case of side note (side note is of the from '[<note>]' and could include multiple words
        return [tkn]
    else:
        return tkn.split(' ')
transcripts[config.CONTENT] = transcripts[config.CONTENT].progress_apply(separate_jumped_tkns)

print('Explode ofter splitting words')
transcripts = transcripts.explode(config.CONTENT)

#%%
# formulate table and save
print('Formulate and save')
transcripts = transcripts[[config.FNAME, config.INTERVIEW_ID, config.TAPE_ID, config.SECTION_ID, config.CONTENT_ID, config.TIME, config.CONTENT]]

transcripts[config.INTERVIEW_ID] = transcripts[config.INTERVIEW_ID].progress_apply(int)
transcripts[config.TAPE_ID] = transcripts[config.TAPE_ID].progress_apply(int)
transcripts[config.SECTION_ID] = transcripts[config.SECTION_ID].progress_apply(int)

transcripts = transcripts.sort_values(by=[config.INTERVIEW_ID, config.TAPE_ID, config.TIME])

transcripts.to_pickle(save_dir + os.sep + 'contents.pkl')

#%%
transcripts = pd.read_pickle(save_dir + os.sep + 'contents.pkl')
# full_interviews = transcripts.groupby('interview_id')['content'].apply(lambda x: ' '.join(x))
transcripts['prev_info'] = transcripts[['interview_id', 'section_id', 'content_id']].shift(-1).progress_apply(lambda x: (x['interview_id'], x['section_id'], x['content_id']), axis=1)

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
transcripts['sep_tkn'] = transcripts[['interview_id', 'section_id', 'content_id', 'prev_info']].progress_apply(get_sep, axis=1)

#%%
for i, row in tqdm(transcripts.iterrows(), total=len(transcripts)):
    txt = ''
    if len(txt) > 0 and row['sep_tkn'] == '<new_doc>':
        with open(r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews' + os.sep + str(row['interview_id']) + '.txt', 'w', encoding='utf-8') as f:
            f.write(txt)
        txt = ''
    else:
        txt += row['sep_tkn']

    txt += row['content']

#%%
for i, row in tqdm(full_interviews.reset_index().iterrows(), total=len(full_interviews)):
    if
    with open(r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews' + os.sep + str(t['interview_id']) + '.txt', 'w', encoding='utf-8') as f:
        f.write(t['content'])