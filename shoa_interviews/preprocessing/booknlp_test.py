from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "big"
}

booknlp = BookNLP("en", model_params)

# Input file to process
input_file = r'C:/Data/shoa_dataset/Martha_transcripts/full_interviews/54374.txt'

# Output directory to store resulting files in
output_directory = r'C:/Data/shoa_dataset/Martha_transcripts/full_interviews_booknlp_entities'

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id = "bartleby"

booknlp.process(input_file, output_directory, book_id)