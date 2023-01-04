import streamlit as st
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel
import sqlite3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load the model and tokenizer
tokenizer_decoder = AutoTokenizer.from_pretrained("tscholak/2jrayxos")
model_decoder = AutoModelForSeq2SeqLM.from_pretrained("tscholak/2jrayxos")

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encoder_decoder_1(query_sentence, table_names, tokenizer, model, cursor):
    # Tokenize query sentence, table names
    query_sentence_encoded = tokenizer(
        [query_sentence], padding=True, truncation=True, return_tensors='pt')
    table_names_encoded = tokenizer(
        table_names, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings for query sentence, table names
    with torch.no_grad():
        query_sentence_output = model(**query_sentence_encoded)
        table_names_output = model(**table_names_encoded)

    # Perform pooling for query sentence, table names
    query_sentence_embedding = mean_pooling(
        query_sentence_output, query_sentence_encoded['attention_mask'])
    table_names_embeddings = mean_pooling(
        table_names_output, table_names_encoded['attention_mask'])

    # Normalize embeddings for query sentence, table names
    query_sentence_embedding = F.normalize(
        query_sentence_embedding, p=2, dim=1)
    table_names_embeddings = F.normalize(table_names_embeddings, p=2, dim=1)

    # Find the most similar table names by computing the cosine similarity between the query sentence embedding and the table names
    # Find the most similar table names by computing the cosine similarity between the query sentence embedding and the table names embeddings
    cosine_similarities_tables = torch.nn.functional.cosine_similarity(
        query_sentence_embedding, table_names_embeddings, dim=1)
    most_similar_table_names_indices = cosine_similarities_tables.argsort(
        descending=True)
    most_similar_table_names = [table_names[i]
                                for i in most_similar_table_names_indices]

    # Find the index of the highest matching table name by finding the maximum value in the list of cosine similarities for the table names
    max_similarity_table_index = cosine_similarities_tables.argmax()

    # Get the highest matching table name by using the index obtained above
    highest_matching_table_name = table_names[max_similarity_table_index]

    # Find the column names of the highest matching table by querying the database
    cursor.execute(f"PRAGMA table_info({highest_matching_table_name});")
    highest_matching_table_column_names = [
        column_info[1] for column_info in cursor.fetchall()]

    highest_matching_table_column_names = ", ".join(
        highest_matching_table_column_names)

    highest_matching_table_column_names_encoded = tokenizer_decoder(
        highest_matching_table_column_names, padding=True, truncation=True, return_tensors='pt', max_length=1024)

    # Generate a summary of the table using the column names as input to the decoder
    summary = model_decoder.generate(**highest_matching_table_column_names_encoded,
                                    num_beams=5,
                                    length_penalty=1.0,
                                    early_stopping=True)

    # Convert the summary from a tensor to a list of tokens
    summary_tokens = [
        tokenizer_decoder.decode(g, skip_special_tokens=True) for g in summary]

    # Concatenate the summary into a single string
    summary_text = " ".join(summary_tokens[0])

    return summary_text

# Create a connection to the database
connection = sqlite3.connect("database.db")
cursor = connection.cursor()

# Get the list of table names in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
all_table_names = [table_info[0] for table_info in cursor.fetchall()]

# Create the main GUI using Streamlit
st.title("SQLite Database Summary Generator")

query_sentence = st.text_input("Enter your query:")
table_names = st.text_input("Enter the names of the tables (comma-separated):")

if st.button("Generate Summary"):
    summary = encoder_decoder_1(query_sentence, table_names, tokenizer, model, cursor)
    st.success(summary)
