import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"C:\Users\Itamar Trainin\Desktop\lexical-acolyte-343718-242b49d379b2.json"

import pandas as pd

from google.cloud import language_v1
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

num_threads = 25
total_num_files = 100

work_dir = r'C:\Data\shoa_dataset\Martha_transcripts\full_interviews'

pbar = tqdm(total=total_num_files)

client = language_v1.LanguageServiceClient()
type_ = language_v1.Document.Type.PLAIN_TEXT
language = "en"
encoding_type = language_v1.EncodingType.UTF8

def ent_ext(txt):
    document = {"content": txt, "type_": type_, "language": language}
    response = client.analyze_entities(request = {'document': document, 'encoding_type': encoding_type})

    entities = []
    for entity in response.entities:
        value = entity.name 
        type = language_v1.Entity.Type(entity.type_).name
        salience = entity.salience
        metadata = []
        for metadata_name, metadata_value in entity.metadata.items():
            metadata.append({
                'name': metadata_name,
                'value': metadata_value
            })

        mentions = []
        for mention in entity.mentions:
            mentions.append({
                'content': mention.text.content,
                'begin_offset': mention.text.begin_offset,
                'type': language_v1.EntityMention.Type(mention.type_).name
            })

        entities.append({
            'value': value,
            'type': type,
            'salience': salience,
            'metadata': metadata,
            'mentions': mentions
        })

    return pd.DataFrame.from_dict(entities)


def thread_func(file):
    # print(f'processing: {file}')
    with open(file, 'r', encoding='utf-8') as f:
        txt = f.read()
        res = ent_ext(txt)
        res['fname'] = pd.Series([os.path.basename(file)]*len(res))
        pbar.update(1)
        return res

#%%
files = [work_dir + os.sep + fname for fname in os.listdir(work_dir)][:total_num_files]

pool = ThreadPool(num_threads)
result = pool.map(thread_func, files)

entities = pd.concat(result)

entities.to_pickle(r'C:\Data\shoa_dataset\Martha_transcripts\entities\google\entities.pkl')

#%%
entities = pd.read_pickle(r'C:\Data\shoa_dataset\Martha_transcripts\entities\google\entities.pkl')

#%%
entities['num_mentions'] = entities['mentions'].apply(len)

person_entities = entities[entities['type'] == 'PERSON'].reset_index(drop=True)
person_entities.index = person_entities['fname']

max_count_per_file = person_entities.groupby('fname')['num_mentions'].max()
person_entities['max_mentions_in_file'] = person_entities['fname'].apply(lambda x: max_count_per_file[x])

person_entities = person_entities.sort_values(by=['max_mentions_in_file', 'fname', 'num_mentions'], ascending=[False, True, False])

#%%
cross_files_entity_counts = person_entities.groupby('fname')['value'].apply(set).explode('value').reset_index()['value'].value_counts()
cross_files_entity_counts.to_excel(r'C:\Data\shoa_dataset\Martha_transcripts\entities\google\cross_files_entity_counts.xlsx')

#%%
t = person_entities.loc['20566.txt']