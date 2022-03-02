import json
import pandas as pd

#%%
with open(r"C:\Data\shoa_dataset\Martha_transcripts\topics\segments_with_topics.json", 'r', encoding='utf-8') as f:
    segments = json.load(f)

#%%
segments_df = pd.DataFrame(segments, columns=['topic', 'segment'])

#%%
segments_df.to_excel(r"C:\Data\shoa_dataset\Martha_transcripts\topics\segments_with_topics.xlsx", encoding='utf-8-sig')
segments_df.to_pickle(r"C:\Data\shoa_dataset\Martha_transcripts\topics\segments_with_topics.pkl")

#%%
topics_distrib = segments_df['topic'].value_counts()