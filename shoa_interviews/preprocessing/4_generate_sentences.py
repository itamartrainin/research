import os
import re
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from shoa_interviews.preprocessing import config
from datetime import timedelta

#%%
working_dir = r'C:\Data\shoa_dataset\Martha_transcripts\outputs'
save_dir = working_dir

INTERVIEW_ID_COL = 'IntCode'
TOPICS_COL = 'IndexedTermLabels'
START_TIME_COL = 'InTimeCode'
END_TIME_COL = 'OutTimeCode'

assert os.path.isdir(working_dir)
assert os.path.isdir(save_dir)

#%%
# Reading files.
print('Reading files.')
transcripts = pd.read_pickle(working_dir + os.sep + 'contents.pkl')
topics = pd.read_pickle(working_dir + os.sep + 'topics.pkl')

#%%
# Transform words into sentences
print('Transform words into sentences')

# End of sentence is a word with !/?/. in it.
END_OF_SENTENCE_REG = re.compile(r'[!?.]')
# Speaker if the token has ':' in it.
SPEAKER_TOKEN_REG = re.compile(r':')

transcripts['is_sent_end'] = transcripts['content'].progress_apply(lambda content: END_OF_SENTENCE_REG.search(content) is not None)
transcripts['is_speaker_tkn'] = transcripts['content'].progress_apply(lambda content: SPEAKER_TOKEN_REG.search(content) is not None)
transcripts['is_last_tkn_in_interview'] = (transcripts['interview_id'] != transcripts['interview_id'].shift(-1)) #this int.id != next int.id
transcripts['is_last_tkn_in_tape'] = (transcripts['tape_id'] != transcripts['tape_id'].shift(-1)) #this tape.id != next tape.id

# It is an end of sentence if it has an end of sentence marking or the next token is a 'speaker' token
transcripts['is_sent_end'] = transcripts['is_sent_end'] | transcripts['is_speaker_tkn'].shift(-1) |\
                             transcripts['is_last_tkn_in_interview'] | transcripts['is_last_tkn_in_tape']

#%%
# Enumerate sentences
print('Enumerate sentences')

sc = 0
output = []
transcripts[config.SENTENCE_ID] = None
for t in tqdm(transcripts['is_sent_end']):
    output.append(sc)
    if t:
        sc += 1
transcripts[config.SENTENCE_ID] = output

del output

#%%
# Group the words into sentences by sentence number
print('Group the words into sentences by sentence number')

grouper = transcripts.groupby(config.SENTENCE_ID)
sentences = grouper.agg(list)

#@TODO !!! REMOVE THIS !!!
sentences_save = sentences

del transcripts

#%%
# Remove empty sentences
print('Remove empty sentences')

sentences = sentences[sentences['content'].progress_apply(lambda content: len(content) > 0)]

#%%
# Add speaker to each sentence
print('Add speaker to each sentence')

current_speaker = None
speakers = []
for i, row in tqdm(sentences.iterrows(), total=len(sentences)):
    if  row['is_speaker_tkn'][0]:
        # Extract the speaker (Change the current speaker)
        current_speaker = row['content'][0].split(':')[0]
        # Remove the speaker from sentence
        row['content'] = row['content'][1:]
    speakers.append(current_speaker)
sentences[config.SPEAKER] = speakers

#%%
# Join tokens into sentences
print('Join tokens into sentences')

sentences['sentence'] = sentences['content'].progress_apply(lambda toks: ' '.join(toks))

#%%
# Convert time to timedelta
print('Convert time to timedelta')

start_end_times = [[timedelta(milliseconds=min(times)), timedelta(milliseconds=max(times))] for times in sentences['time'].tolist()]
sentences[['start_time', 'end_time']] = pd.DataFrame(start_end_times, index=sentences.index)

#%%
sentences = sentences.drop(columns=['fname'])
sentences['interview_id'] = sentences['interview_id'].progress_apply(lambda x: x[0])
sentences['tape_id'] = sentences['tape_id'].progress_apply(lambda x: x[0])
sentences['section_id'] = sentences['section_id'].progress_apply(lambda x: x[0])

#%%
sentences.sort_values(by=['interview_id', 'tape_id', 'start_time', 'end_time'])
topics.sort_values(by=['IntCode', 'InTapenumber', 'OutTapenumber', 'InTimeCode', 'OutTimeCode'])

#%%
sentences = sentences.set_index('interview_id')
topics = topics.set_index('IntCode')

#%%
#@TODO !!! REMOVE THIS !!!
sentences_save = sentences
topics_save = topics

#%%
def time_ranges_overlap(a, b):
    # Checks if range a overlaps range b
    return (a[0] <= b[1]) and (a[1] >= b[0])

def is_before(a, b):
    # Check if range a completely before (no-overlap + latter in time) range b
    return a[1] < b[0]

def is_after(a, b):
    # Check if range a completely after (no-overlap + latter in time) range b
    return b[1] < a[0]

#%%
interview_ids = list(set(topics.index))

#%%

sentences['segment_id'] = None

for interview_id in tqdm(interview_ids):
    interview_topics = topics.loc[interview_id]
    interview_sentences = sentences.loc[interview_id]

    seg_id = 0
    topics_index = 0
    is_currently_nan_seg = False

    for sentence in interview_sentences.iterrows():
        topic_time = (topics[topics_index]['InTimeCode'], topics[topics_index]['OutTimeCode'])
        sentence_time = (sentence['start_time'], sentence['end_time'])

        if is_before(topic_time, sentence_time):
            # We are at a nan section
            if not is_currently_nan_seg:
                # Change of segment
                is_currently_nan_seg = True
                seg_id += 1
        elif is_after(topic_time, sentence_time):
            pass
        # else:
        #     pass


    sentences_index = 0

    while topics_index < len(interview_topics) and sentences_index < len(interview_sentences):
        pass


#%%


###################################


#%%
sentences['segment_id'] = None
interview_ids = list(set(topics.index))
for interview_id in tqdm(interview_ids):
    interview_topics = topics.loc[interview_id].reset_index('IntCode')
    interview_sentences = sentences.loc[interview_id]
    for _, topic in interview_topics.iterrows():
        overlappings = interview_sentences[
            (interview_sentences['start_time'] <= topic['OutTimeCode']) & (interview_sentences['end_time'] >= topic['InTimeCode'])
        ]
        sentences.loc[overlappings.index, 'segment_id'] = topic['SegmentNumber']


    break
        # for i, sentence in interview_sentences.iterrows():
        #     if time_ranges_overlap(sentence['start_time'], sentence['end_time'], topic['InTimeCode'], topic['OutTimeCode']):
        #         sentence['segment_id'] = seg_id
        #     elif has_passed_range(sentence['start_time'], sentence['end_time'], topic['InTimeCode'], topic['OutTimeCode']):
        #         break



interview_sentences = sentences.loc[interview_id]
interview_start_times = list(interview_topics['start_time'])
interview_end_times = list(interview_topics['end_time'])
sentences_times = zip(interview_start_times, interview_end_times)

#%%
t = sentences[sentences['fname'].apply(lambda x: len(set(x)) > 1)] # --> same interview different tape number
t = sentences[sentences['interview_id'].apply(lambda x: len(set(x)) > 1)] # --> none
t = sentences[sentences['tape_id'].apply(lambda x: len(set(x)) > 1)] # ?
t = sentences[sentences['section_id'].apply(lambda x: len(set(x)) > 1)] # ?

topics_start_times = list(interview_topics['InTimeCode'])
topics_end_times = list(interview_topics['OutTimeCode'])
topics_times = zip(topics_start_times, topics_end_times)

#%%
sentences = sentences.set_index('interview_id')
topics = topics.set_index('IntCode')






#%%
topics_grouped_by_interview_id = topics.groupby(INTERVIEW_ID_COL)

#%%
num_segments_per_interview = topics_grouped_by_interview_id['SegmentNumber'].max()

#%%
interview_ids = list(set(transcripts['interview_id']))

#%%
# set transcripts index to be the interview number
transcripts = transcripts.set_index('interview_id')
sentences = sentences.set_index('')
topics = topics.set_index('IntCode')

#%%
# for interview_id in tqdm(interview_ids):
#     interview_topics = topics.loc[interview_id]
#     interview_sentences = senteces




#%%
sentences.to_pickle(save_dir + os.sep + 'sentences.pkl')




