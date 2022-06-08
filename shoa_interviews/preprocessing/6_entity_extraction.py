#%%
import os
import pickle

import pandas as pd
import stanza
import pickle

from multiprocessing.pool import Pool, ThreadPool

from tqdm import tqdm

tqdm.pandas()

#%%
work_dir = r'C:/Data/shoa_dataset/Martha_transcripts/full_interviews'
save_dir = r'C:/Data/shoa_dataset/Martha_transcripts/full_interviews_entities'

#%%
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

total_num_files = 80
files = [fname for fname in os.listdir(work_dir)][:total_num_files]
for file in tqdm(files):
    with open(work_dir + os.sep + file, 'r', encoding='utf-8') as f:
        content = f.read()
        doc = nlp(content)
        ent_pairs = [(ent.text, ent.type) for ent in doc.ents]
        pickle.dump((ent_pairs, doc), open(save_dir + os.sep + file[:-4] + '.pkl', 'wb'))

#%%
# num_processes = 8
num_processes = 2
num_threads = 10


process_chunk_size = int(len(files) / num_processes)
process_files = [files[i*process_chunk_size:(i+1)*process_chunk_size] for i in range(num_processes)]

def thread_func(params):
    nlp, file = params
    print(f'Processing: {file}')
    with open(work_dir + os.sep + file, 'r', encoding='utf-8') as f:
        content = f.read()
        doc = nlp(content)
        ent_pairs = [(ent.text, ent.type) for ent in doc.ents]
        pickle.dump((ent_pairs, doc), open(save_dir + os.sep + file[:-4] + '.pkl', 'wb'))

def process_func(files):
    print('hi')
    # nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
    #
    # thread_chunk_size = int(len(files) / num_threads)
    # thread_files = [files[i * thread_chunk_size:(i + 1) * thread_chunk_size] for i in range(num_threads)]
    #
    # params = list(zip([nlp]*len(thread_files), thread_files))
    #
    # thread_pool = ThreadPool(num_threads)
    # thread_pool.map(thread_func, params)

pool = Pool(num_processes)
pool.map(process_func, process_files)

#%%
all_entities = pd.DataFrame([[file, pickle.load(open(save_dir + os.sep + file, 'rb'))[0]] for file in tqdm(os.listdir(save_dir))], columns=['fname', 'entities'])
all_entities = all_entities.explode('entities').reset_index(drop=True)
all_entities[['value', 'tag']] = pd.DataFrame(all_entities['entities'].tolist())
all_entities = all_entities.drop(columns='entities')

person_entities = all_entities[all_entities['tag'] == 'PERSON']

person_counts_per_doc = person_entities.groupby(by=['fname', 'value'])['value'].size()
person_counts_per_doc = person_counts_per_doc.reset_index(name='count')

max_count_per_file = person_counts_per_doc.groupby('fname')['count'].max()
person_counts_per_doc['max_count_in_file'] = person_counts_per_doc['fname'].apply(lambda x: max_count_per_file[x])

person_counts_per_doc = person_counts_per_doc.sort_values(by=['max_count_in_file', 'fname', 'count'], ascending=[False, True, False])

cross_files_entity_counts = person_counts_per_doc['value'].value_counts()

#%%
from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "big"
}

booknlp = BookNLP("en", model_params)
output_directory = r'C:/Data/shoa_dataset/Martha_transcripts/full_interviews_booknlp_entities'

total_num_files = 80
files = [fname for fname in os.listdir(work_dir)][:total_num_files]
for file in tqdm(files):
    booknlp.process(work_dir + os.sep + file, output_directory, file)

