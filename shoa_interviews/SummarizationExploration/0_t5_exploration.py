import pandas as pd
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

#%%
model = T5Model.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

#%%
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

#%%
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')

#%%
max_input = model.config.n_positions
print(f'Max input size: {max_input}')

#%%
input_text = """INTERVIEWER: Were you at the square when the picking was done? SUBJECT: No. No. I was at home. I remember someone coming and telling my grandmother that the despair and the disbelief. And I remember an aunt, my grandmother's actually my father's aunt, grabbing me by the hand and taking me shopping just to get us out of there, I guess. My sister was younger, so maybe she wasn't as involved with this. But. And I remember going to the market, which was opened only a certain few hours to the Jewish people. And I remember coming back with a basket of eggs. Maybe I remember it because my aunt tells me this story often. She is still living. And I guess there was so much horror in the house, in our house where we live now, in that confusion, I stepped in the basket of eggs and ruined that shopping. So that's when my uncle and other uncle who was a doctor and he was the head of the ambulatorium, which was sort of like a hospital in Yugoslavia for all the Jewish people. Decided that maybe there was we would have to get out and leave towards Italy, somehow make our way to Italy. Of course, I don't remember all of these details. Because we were not made part of this. We were very small children. However, I know my mother was wearing that awful yellow tag on her arm. And there were curfews and there were awful daily things that we went through."""

#%%
tokenized_text = tokenizer.encode('summarize: ' + input_text, truncation='only_first', return_tensors='pt')
truncated_input = tokenizer.decode(tokenized_text[0], skip_special_tokens=True)

#%%
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=250,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("Summarized text:\n", output)

#%%
# Infinite Progressive Summarization
summ = input_text
summ_len = len(tokenizer.encode(summ))
i = 0
while summ_len > 1:
    tokenized_text = tokenizer.encode('summarize: ' + summ, truncation='only_first', return_tensors='pt')
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=int(0.8*summ_len),
                                        max_length=int(0.9*summ_len),
                                        early_stopping=True)

    summ = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Summarized text (iter #{i}), summ length: {summ_len}:\n{summ}\n")
    summ_len = len(tokenizer.encode(summ))
    i += 1
