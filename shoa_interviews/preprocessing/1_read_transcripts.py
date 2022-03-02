import os
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from shoa_interviews.preprocessing import config

#%%
working_dir = r'C:\Data\shoa_dataset\Martha_transcripts\transcripts'
save_dir = r'C:\Data\shoa_dataset\Martha_transcripts\outputs'

assert os.path.isdir(working_dir)
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
    """
    Iterate over the files in dir and read them into a dataframe
    :param dir: folder with transcripts
    :return: dataframe with transcripts xmls
    """
    files = []
    for fname in tqdm(os.listdir(dir)):
        file_meta = get_file_meta(fname)
        with open(dir + os.sep + fname, 'r') as f:
            files.append([fname, file_meta[0], file_meta[1], f.read()])
    return pd.DataFrame(files, columns=[config.FNAME, config.INTERVIEW_ID, config.TAPE_ID, config.TRANS_XML])

#%%
print('Reading Files')
# Read the transcripts
transcripts_files = read_files(working_dir)

# Save transcripts xmls in dataframe.
transcripts_files.to_pickle(save_dir + os.sep + 'transcripts.pkl')
