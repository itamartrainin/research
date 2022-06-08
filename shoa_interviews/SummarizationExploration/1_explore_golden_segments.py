import os
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from spacy.tokens import Doc
from spacy.vocab import Vocab
from transformers import T5Tokenizer

tqdm.pandas()

#%%
docs_dir = r"C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\spacy_docs"
save_dir = r"C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\topic_to_text"

#%%
"""
    Load Doc
    one_doc = Doc(Vocab()).from_disk(r"C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\doc_101")

    Access the full document
    one_doc.doc

    Access the segments
    one_doc.spans['segments']
    
    Access start and end index of the span
    one_doc.spans['segments'][0].start
    one_doc.spans['segments'][0].end
    
    Access topics of a segment
    one_doc.user_data[('._.', 'topics', None, None)]
    OR
    one_doc. //set extension
"""

docs = [Doc(Vocab()).from_disk(docs_dir + os.sep + fname) for fname in os.listdir(docs_dir)]

#%%
segments = pd.DataFrame([[doc_ix, seg_ix, topic, str(seg)] for doc_ix, doc in enumerate(docs) for seg_ix, (seg, topic) in enumerate(zip(doc.spans['segments'], doc.user_data[('._.', 'topics', None, None)]))], columns=['doc_ix', 'seg_ix', 'topic', 'content'])

#%%
tokenizer = T5Tokenizer.from_pretrained('t5-base')
segments['num_tokns'] = segments['content'].progress_apply(lambda x: len(tokenizer(x).input_ids))
mean_num_tokns = segments['num_tokns'].mean()

#%%
plt.hist(segments['num_tokns'], bins=25)
plt.axvline(mean_num_tokns, color='red')
plt.title('Segment Length Distribution')
plt.xlabel('# tokens')
plt.ylabel('# segments')
plt.text(mean_num_tokns+20, 155, 'mean-length', rotation=-90)

#%%
max_num_tokns = 512
split_token_ix = 5

#%%
# Split all segments that are longer than max_seg_size
def get_cut_index(encoded, max_num_tokns, split_token_ix):
    truncated_encoded = encoded[:max_num_tokns - 1]
    if split_token_ix in truncated_encoded:
        cut_off_index = len(truncated_encoded) - truncated_encoded[::-1].index(split_token_ix) - 1
    else:
        print('truncation token was not found')
        cut_off_index = max_num_tokns - 1
    return cut_off_index

def split_segment(row, max_num_tokns, split_token_ix):
    # Split segments longer than max_num_tokns by finding the position of the nearest '.'
    parts = []
    to_split = row['content']
    while True:
        encoded = tokenizer.encode(to_split)

        cut_off_index = get_cut_index(encoded, max_num_tokns, split_token_ix)

        # Cut
        part_1 = encoded[:cut_off_index + 1]
        part_2 = encoded[cut_off_index + 1:-1]

        # Append part 1
        if len(part_1) < max_num_tokns:
            parts.append([row['doc_ix'], tokenizer.decode(part_1), None])
        else:
            print('here')

        # If append part 2 only if it is shorter than max_num_tokns and exit.
        if len(part_2) <= max_num_tokns:
            parts.append([row['doc_ix'], tokenizer.decode(part_2), None])
            break
        else:
            to_split = tokenizer.decode(part_2)

    return parts

split_segments = []
for i, row in tqdm(segments.iterrows(), total=len(segments)):
    if row['num_tokns'] >= max_num_tokns:
        parts = split_segment(row, max_num_tokns, split_token_ix)
        split_segments += parts
    else:
        split_segments.append([row['doc_ix'], row['content'], None])

split_segments = pd.DataFrame(split_segments, columns=['doc_ix', 'content', 'num_tokns'])
split_segments['num_tokns'] = split_segments['content'].progress_apply(lambda x: len(tokenizer(x).input_ids))

#%%
# Join Segments
merged_segments = []
i = 0
while i < len(split_segments):
    row = split_segments.iloc[i]

    cum_text = row['content']
    cum_length = row['num_tokns']
    cum_count = 1
    for _, other_row in split_segments[i+1:].iterrows():
        if row['doc_ix'] != other_row['doc_ix'] or cum_length + other_row['num_tokns'] - 1 > max_num_tokns: # -1 for duplicate </s>
            break
        else:
            cum_text += ' ' + other_row['content']
            cum_length += other_row['num_tokns'] - 1
            cum_count += 1

    merged_segments.append([row['doc_ix'], cum_text, None])
    i += cum_count

reduced_segments = pd.DataFrame(merged_segments, columns=['doc_ix', 'content', 'num_tokns'])
reduced_segments['num_tokns'] = reduced_segments['content'].progress_apply(lambda x: len(tokenizer(x).input_ids))
reduced_mean_num_tokns = reduced_segments['num_tokns'].mean()

#%%
print(reduced_segments['num_tokns'].describe())
print(f'Number of segments shorter than {max_num_tokns} tkns: '
      f'{len(reduced_segments[reduced_segments["num_tokns"]<=max_num_tokns])}/{len(reduced_segments)}'
      f' ({int(100*len(reduced_segments[reduced_segments["num_tokns"]<=max_num_tokns])/len(reduced_segments))}%)')


#%%
plt.hist(reduced_segments['num_tokns'], bins=25)
plt.axvline(reduced_mean_num_tokns, color='red')
plt.title('Segment Length Distribution (after resizing)')
plt.xlabel('# tokens')
plt.ylabel('# segments')
plt.text(reduced_mean_num_tokns, 155, 'mean-length', rotation=-90)

#%%
reduced_segments.sample(frac=1).to_excel(save_dir + os.sep + 'segments.xlsx')

#%%
# Save segments
# def format_segment(row):
#     # text = '#' * 10 + f' {row["seg_ix"]} / {row["topic"]} ' + '#' * 10 + '\n'
#     text = '#' * 20 + '\n'
#     text += row['content'] + '\n'
#     text += '#' * 10 + f' SUMMARY ' + '#' * 10 + '\n'
#     text += '\n\n\n'
#     return text
#
# reduced_segments['seg_txt'] = reduced_segments.apply(format_segment, axis=1)
# documents = reduced_segments.groupby('doc_ix')
# for doc in documents:
#     with open(save_dir + os.sep + str(doc[0]) + '.txt', 'w', encoding='utf-8') as f:
#         f.write('\n'.join(doc[1]['seg_txt']))


#%%
segments = segments[segments['topic'] != 'NO_TOPIC']
segments['content_len'] = segments['content'].apply(len)
segments = segments.sort_values(by=['topic', 'content_len'], ascending=[True, False])

