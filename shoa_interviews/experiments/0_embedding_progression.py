#%%
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

#%%
model = SentenceTransformer('all-mpnet-base-v2')

#%%
# with open('', 'r', encoding='utf-8') as f:
#     txt = f.read()

#%%
txt = """But this-- you have to get inside Switzerland. Well, we go up to the station and-- and-- and quickly tried to get on this train so that the Germans won't bother us since you have no papers to get out. I get on the train. I have to have this elderly lady off the-- on-- on the train, you know? It's that train that's moving.
I help her up. She gets up. And as we come by the Swiss, I'm still hanging on the outside. And they pull me off. Now if I had been alone, maybe I could have made it. I don't know. Anyway, the old-- the lady was on the-- on the train. And I was not.
INTERVIEWER 2: The Germans pulled you off or?
SUBJECT: The Swiss pulled me off. And, uh--
INTERVIEWER 1: But you weren't in Switzerland yet, no?
SUBJECT: No, on-- on the border. But this-- that-- the-- the two stations located at a-- like-- like, practically the same station. It's 100 yards. The train moves. And-- and-- and-- and by the time she got on, I couldn't make it inside the train.
INTERVIEWER 2: They pulled you off and checked the passport?
SUBJECT: They pulled me off. And now-- this was not the Germans, now. I'm talking to the Swiss. And-- and-- and the Swiss say, well, what are you trying to do? I said, well, I'm trying to leave Germany because I can't live there.
And do you have a Swiss visa? I said, no. I can't let you in. And I pleaded with them for 10, 15 minutes. Told them I had an affidavit. And I'm only waiting for my quota number to go to the United States. I showed them a copy of my affidavit.
There was absolutely no way they would let me in. And, uh, and I said, you know, what are my options? I mean, I have to return to Germany? The Gestapo's going to arrest me. Uh, I'm sorry. But we have already admitted so many of, uh, of, uh, you know, refugees. We are not permitting anybody anymore without a visa.
And, uh, I had to go back. And, uh, actually, I'm sorry. This is now 40 years, and I had forgotten. It was the Germans that pulled me off the train. The Germans pulled me off the train. I have to correct this. You have to have to-- when you edit it, you have to correct-- I've forgotten.
This is-- this is-- the Germans pulled me off the train. And the-- this-- and-- and they-- they started interrogating me. What was I doing? I said, I'm Jew-- I'm a Jew from Vienna. have no future here, as you know. I want to emigrate to America. But I can no longer wait. I have nothing to do. I'm-- I'm going crazy. I want to leave.
They-- they examined my papers. Uh, my passport was in order. How much money do you have? I said 10 marks. That was all I was allowed to take. If I had 15, they would have arrested me. I had 10 marks.
INTERVIEWER 1: Did they search you?
SUBJECT: They searched me, but I had no more money left. Uh, and they said, OK, you want to leave? Ask the Swiss. That's how it was. The woman was on the train. I asked the Swiss.
So I know why I had to do it this way because I could not buy a ticket to go to Switzerland. I could not officially buy the ticket to go. So we had to get on the train even without the OK of the Germans, OK?
I got on the train. But I didn't have time to get on. They pulled me off. But the German was fine. He said, you know, fine. As long as I didn't break the laws and I had a pass, but he didn't care. You go talk to the Swiss. The Swiss turned me down.
"""

#%%
sentences = re.split(r"[.!?] |\n", txt)

#%%
embeddings = model.encode(sentences)

#%%
embeddings_2d = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embeddings)

#%%
for i in range(1, embeddings_2d.shape[0]):
    plt.clf()
    plt.cla()
    plt.close()
    print(f'{i} - {sentences[i-1]}')
    sns.scatterplot(x=embeddings_2d[:i,0], y=embeddings_2d[:i,1])
    plt.show()
    time.sleep(5)

#%%
i = 0

#%%
plt.clf()
plt.cla()
plt.close()
for i in range(i+1):
    print(f'{i} - {sentences[i]}')
sns.scatterplot(x=embeddings_2d[:i+1, 0], y=embeddings_2d[:i+1, 1])
plt.xlim(-6,12)
plt.ylim(-10,6)
plt.show()
i+=1

