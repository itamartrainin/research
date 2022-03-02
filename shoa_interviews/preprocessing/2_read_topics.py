import os
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from shoa_interviews.preprocessing import config
from datetime import datetime

#%%
working_dir = r'C:\Data\shoa_dataset\Martha_transcripts\topics'
save_dir = r'C:\Data\shoa_dataset\Martha_transcripts\outputs'

assert os.path.isdir(working_dir)
assert os.path.isdir(save_dir)

#%%
# Preprocess Topics
topics = pd.read_csv(working_dir + os.sep + 'topics.csv', encoding='utf-8', engine='python')

#%%
topics = topics[~topics[config.TOPICS_COL].isna()]
topics[config.TOPICS_COL] = topics[config.TOPICS_COL].progress_apply(lambda topics: topics.split(';'))

#%%
def time_to_seconds(t):
    return datetime.strptime(t, '%H:%M:%S:%f') - datetime.strptime('00:00:00:00', '%H:%M:%S:%f')

topics[config.START_TIME_COL] = topics[config.START_TIME_COL].progress_apply(time_to_seconds)
topics[config.END_TIME_COL] = topics[config.END_TIME_COL].progress_apply(time_to_seconds)

#%%
topics_sep = topics.explode(config.TOPICS_COL)
topics_sep[config.TOPICS_COL] = topics_sep[config.TOPICS_COL].apply(lambda x: x.strip())
topics_counts = topics_sep[config.TOPICS_COL].value_counts()

#%%
topics = topics.sort_values(by=['IntCode', 'InTapenumber', 'OutTapenumber', 'InTimeCode', 'OutTimeCode'])

#%%
# Check that there are no overlaps in times
topics['next_InTimeCode'] = topics['InTimeCode'].shift(-1)
topics['next_OutTimeCode'] = topics['OutTimeCode'].shift(-1)
topics['next_InTapenumber'] = topics['InTapenumber'].shift(-1)
topics['next_IntCode'] = topics['IntCode'].shift(-1)
overlaps = topics[topics.apply(lambda row: row['OutTimeCode'] > row['next_InTimeCode'] and
                                    row['InTapenumber'] == row['next_InTapenumber'] and
                                    row['IntCode'] == row['next_IntCode'],
                        axis=1)]
assert len(overlaps) == 0

# %%
#
#
# %%
# topics = topics.sort_values(by=['IntCode', 'InTapenumber', 'OutTapenumber', 'InTimeCode', 'OutTimeCode'])
#
#%%
topics.to_pickle(save_dir + os.sep + 'topics.pkl')
topics.to_excel(save_dir + os.sep + 'topics.xlsx')
topics_sep.to_pickle(save_dir + os.sep + 'topics_sep.pkl')




#################
#%%
topics = pd.read_pickle(save_dir + os.sep + 'topics.pkl')
