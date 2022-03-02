import os
import re
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from datetime import datetime, timedelta

#%%
#CONSTS
TRANSCRIPT_XML_TAG = 'transcription'
SECTION_XML_TAG = 'p'

# COLUMN NAMES
FNAME = 'fname'
TRANS_XML = 'trans_xml' # The with the original text in xml format
INTERVIEW_ID = 'interview_id'
TAPE_ID = 'tape_id'
SECTION_ID = 'section_id'
CONTENT_ID = 'content_id'
TIME = 'time'
CONTENT = 'content'
SENTENCE_ID = 'sent_id'
SENTENCE_TOKENS = 'sentence_tokens'
SENTENCE = 'sentence'
START_TIME = 'start_time'
END_TIME = 'end_time'
SPEAKER = 'speaker'

# REGEXES
SECTION_REG = re.compile(rf'<{SECTION_XML_TAG}>(.*?)</{SECTION_XML_TAG}>')
CONTENT_REG = re.compile(r"<span m=\'([0-9]+)\'>([^<]+)</span>")
END_OF_SENTENCE_REG = re.compile(r'[!?.]')

#%%
base_dir = r'C:\Data\shoa_dataset\Martha_transcripts'
transcripts_dir = base_dir + os.sep + 'transcripts'
topics_dir = base_dir + os.sep + 'topics'
save_dir = r'C:\Data\shoa_dataset\Martha_transcripts\outputs'

assert os.path.isdir(transcripts_dir)
assert os.path.isdir(save_dir)

#%%
def get_file_meta(filename):
    """
    Extract the meta-data of the file from the name
    :param filename: A filename in the format: [interview_number].[tape_number].[file_type]
    :return: A tuple: ([interview_number, tape_number]) as integers
    """
    t = filename.split('.')
    return int(t[0]), int(t[1])

def read_files(dir):
    files = []
    for fname in tqdm(os.listdir(dir)):
        file_meta = get_file_meta(fname)
        with open(dir + os.sep + fname, 'r') as f:
            files.append([fname, file_meta[0], file_meta[1], f.read()])
    return pd.DataFrame(files, columns=[FNAME, INTERVIEW_ID, TAPE_ID, TRANS_XML])

print('Reading Files')
transcripts_files = read_files(transcripts_dir)

transcripts_files.to_pickle(save_dir + os.sep + 'transcripts.pkl')

#%%
# # Check if all files start with "<transcription>" and end with "</transcription>"
# invalid = transcripts[TRANS_XML].apply(lambda xml: xml[:len(f'<{TRANSCRIPT_XML_TAG}>')] != f'<{TRANSCRIPT_XML_TAG}>' or
#                                                           xml[-len(f'</{TRANSCRIPT_XML_TAG}>'):] != f'</{TRANSCRIPT_XML_TAG}>')
# print(f'Invalid files: {any(invalid)}')
#
# #%%
# # Characters distribution
# characters = transcripts[TRANS_XML].progress_apply(lambda xml: [c for c in xml])
# characters = characters.explode()
# characters = characters.value_counts()
#
# transcripts.to_excel(save_dir + os.sep + 'interviews_char_distribution.xlsx', encoding='utf-8-sig')

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
# Extract Transcript content, that is what is whithing the <transcript>...</transcript>
print('Extracting transcripts contents')
transcripts = transcripts_files
transcripts['transcription'] = transcripts[TRANS_XML].progress_apply(get_transcription_content)
# WARN
# Clean empty transcripts
transcripts = transcripts[transcripts['transcription'] != '']

#%%
# From the transcript, extract the sections (<p>...</p>)
print('Extracting sections')
transcripts['sections'] = transcripts['transcription'].progress_apply(get_sections)
# Convert into a list of sections
transcripts = transcripts.explode('sections')
# WARN
# Clean empty sections
transcripts = transcripts[transcripts['sections'] != '']
# Note the transcript's ID
transcripts = transcripts.rename_axis(SECTION_ID).reset_index()

#%%
# From each sections extract the contents of the section
print('Extract content')
transcripts['contents'] = transcripts['sections'].progress_apply(get_section_contents)
# Convert into a list of contents
transcripts = transcripts.explode('contents')
# Split contents into columns (Time col, Content col)
pairs = [[int(x[0]), x[1]] for x in transcripts['contents'].tolist()]
transcripts[[TIME, CONTENT]] = pd.DataFrame(pairs, index=transcripts.index)
# Note the section ID
transcripts = transcripts.rename_axis(CONTENT_ID).reset_index()

#%%
# Delete temp columns
print('Deleting Columns')
transcripts = transcripts.drop(columns=[TRANS_XML, 'transcription', 'sections', 'contents'])

#%%
# Handle cases where there are multiple words (separated by ' ')
print('Split words')
transcripts[CONTENT] = transcripts[CONTENT].progress_apply(lambda content: content.strip().split(' '))
print('Explode ofter splitting words')
transcripts = transcripts.explode(CONTENT)

#%%
# formulate table and save
print('Formulate and save')
transcripts = transcripts[[FNAME, INTERVIEW_ID, TAPE_ID, SECTION_ID, CONTENT_ID, TIME, CONTENT]]

transcripts[INTERVIEW_ID] = transcripts[INTERVIEW_ID].progress_apply(int)
transcripts[TAPE_ID] = transcripts[TAPE_ID].progress_apply(int)
transcripts[SECTION_ID] = transcripts[SECTION_ID].progress_apply(int)

transcripts = transcripts.sort_values(by=[INTERVIEW_ID, TAPE_ID, TIME])

transcripts.to_pickle(save_dir + os.sep + 'contents.pkl')

#%%
# Transform words into sentences
# End of sentence is a word with !/?/. in it.
transcripts['sent_end'] = transcripts[CONTENT].progress_apply(lambda content: END_OF_SENTENCE_REG.search(content) is not None)

#%%
# Enumerate the sentences
sc = 0
output = []
transcripts[SENTENCE_ID] = None
for t in tqdm(transcripts['sent_end']):
    output.append(sc)
    if t:
        sc += 1
transcripts[SENTENCE_ID] = output
del output

#%%
# Group the words into sentences by sentence number
grouper = transcripts.groupby(SENTENCE_ID)

#%%
sentences = pd.DataFrame([])
sentences[FNAME] = grouper[FNAME].progress_apply(lambda x: list(set(x)))
sentences[INTERVIEW_ID] = grouper[INTERVIEW_ID].progress_apply(lambda x: list(set(x)))
sentences[TAPE_ID] = grouper[TAPE_ID].progress_apply(lambda x: list(set(x)))
sentences[SECTION_ID] = grouper[SECTION_ID].progress_apply(lambda x: list(set(x)))
sentences[CONTENT_ID] = grouper[CONTENT_ID].progress_apply(lambda x: list(set(x)))
sentences[TIME] = grouper[TIME].progress_apply(lambda x: list(x))
sentences[SENTENCE_TOKENS] = grouper[CONTENT].progress_apply(lambda x: list(x))
sentences[START_TIME] = sentences[TIME].apply(lambda t: min(t))
sentences[END_TIME] = sentences[TIME].apply(lambda t: max(t))

#%%
# sent_starts = sentences[SENTENCE_TOKENS].apply(lambda x: x[0])
# sent_starts_with_colns = sent_starts[sent_starts.apply(lambda x: ':' in x and re.search(r'[0-9]', x) is None)].value_counts()

# Special cases
# IK:[PAUSES
# 'Life:

#%%
# Add speaker to each sentence
current_speaker = ''
speakers = []
for sent_toks in tqdm(sentences[SENTENCE_TOKENS]):
    if ':' in sent_toks[0]:
        current_speaker = sent_toks[0].split(':')[0]
    speakers.append(current_speaker)
sentences[SPEAKER] = speakers

#%%
# Remove speaker from sentence and join tokens
sentences[SENTENCE_TOKENS] = sentences[SENTENCE_TOKENS].progress_apply(lambda toks: toks[1:] if ':' in toks[0] else toks)
sentences[SENTENCE] = sentences[SENTENCE_TOKENS].progress_apply(lambda toks: ' '.join(toks))

#%%
# Convert time to timedelta
sentences[START_TIME] = sentences[START_TIME].progress_apply(lambda t: timedelta(milliseconds=t))
sentences[END_TIME] = sentences[END_TIME].progress_apply(lambda t: timedelta(milliseconds=t))

#%%
sentences.to_pickle(save_dir + os.sep + 'sentences.pkl')

#%%
# Preprocess Topics
# topics = pd.read_csv(topics_dir + os.sep + 'index segments for 1000 English Jewish survivor interviews.xlsx', encoding='utf-8', engine='python')
topics = pd.read_csv(topics_dir + os.sep + 'topics.csv', encoding='utf-8', engine='python')

#%%
topics = topics[~topics['IndexedTermLabels'].isna()]
topics['IndexedTermLabels'] = topics['IndexedTermLabels'].progress_apply(lambda topics: topics.split(';'))
topics = topics.explode('IndexedTermLabels')
topics['IndexedTermLabels'] = topics['IndexedTermLabels'].apply(lambda x: x.strip())
topics_counts = topics['IndexedTermLabels'].value_counts()

#%%
def time_to_seconds(t):
    return datetime.strptime(t, '%H:%M:%S:%f') - datetime.strptime('00:00:00:00', '%H:%M:%S:%f')

topics['InTimeCode'] = topics['InTimeCode'].progress_apply(time_to_seconds)
topics['OutTimeCode'] = topics['OutTimeCode'].progress_apply(time_to_seconds)
