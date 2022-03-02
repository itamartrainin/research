import pandas as pd

#%%
df = pd.read_excel(r"C:\Users\Itamar Trainin\Documents\Thesis Research\HUJI\omri_abend\Summarization Model Comparison_v0.xlsx")

#%%
col_name2true_ids = {}
for c in df.columns:
    col_name2true_ids[c] = list(df[df[c]==True]['ID'])

#%%
for k, v in col_name2true_ids.items():
    print(f'{k}:\t\t{v}')
