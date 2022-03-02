#%%
import os
import pandas as pd
from pathlib import Path

#%%
segments = pd.read_excel('C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\joined_segments_tagged\segments.xlsx')
untagged_segments = segments[segments['Summary'].isna()]

#%%
# Save segments
save_dir = r'C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\joined_segments_for_tagging\files'

def format_segment(content):
    return f"$original_text:$\n{content}\n$key_ideas:$\n\n$topic:$\n\n$summary:$\n\n"

for i, row in untagged_segments.iterrows():
    path = save_dir + os.sep + str(int(i/100))
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + os.sep + f'{i}_{row["doc_ix"]}_{row["segment_ix"]}.txt', 'w', encoding='utf-8') as f:
        f.write(format_segment(row['content']))