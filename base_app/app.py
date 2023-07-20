import openai
import streamlit as st
import pandas as pd
from scipy import spatial
import ast

st.title("Ask Divya")
with st.expander("ℹ️ Disclaimer"):
    st.caption(
        "For official immigration advice please consult a certified lawyer"
    )
    
### Initial message ###
message = st.chat_message("assistant", avatar="./base_app/ask_divya.png")
message.write("Hello there, what immigration related question can I help you with today?")
##########################################

### Embedding ###

df = pd.read_csv("base_app/embeddings.csv")
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)


# Embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def prompt_template(prompt):
    prompt_extra = "Address or answer the question by retrieving the information from the following context. If the context does not address the question, say you don't know the answer. Context:"
    strings, relatednesses = strings_ranked_by_relatedness(prompt, df, top_n=3)
    for string in strings:
        prompt_extra += string + "\n"
    return prompt + " " + prompt_extra
        

##########################################

#openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = "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"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="./ask_divya.png"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="🧑🏾"):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑🏾"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="./ask_divya.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": prompt_template(m["content"])} # add prompt template here
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
