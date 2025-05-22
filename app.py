import streamlit as st
from langchain.prompts import PromptTemplate
from llama_cpp import Llama

def llama_response(input_text, number_of_words,blog_type):

    number_of_words = int(number_of_words)

    llm = Llama(model_path = "E:\\blog_generation\\models\\llama-2-fine_tune.gguf",
                n_ctx = 256,
                )
    
    template = """
    Write a Blog for the target audience as {blog_type}, for the topic {input_text} within {number_of_words}.
    """

    prompt = PromptTemplate(input_variable = ["style", "text", "n_words"],
                            template = template)
    
    filled_prompt = prompt.format(blog_type=blog_type, input_text=input_text, number_of_words=number_of_words)

    response = llm(filled_prompt, 
                   max_tokens=number_of_words * 2,  
                   temperature=0.7,                 
                   top_p=0.9,                       
                   top_k=40,                        
                   repeat_penalty=1.1,              
                   )
    print(response)

    return response['choices'][0]['text']

st.set_page_config(
  page_title = "Generate Blogs",
  page_icon = "O_O",
  layout = "centered",
  initial_sidebar_state = 'collapsed'
)

st.title("BLOG GENERATOR")

input_text = st.text_input("Enter the topic for your Blog")

col1,col2 = st.columns([6,6])

with col1:
    number_of_words = st.text_input('Number of Words')
with col2:
    blog_type = st.selectbox('Target your blog to',('Common People', 'Tech Enthusiast','Researchers'),index = 0)

submit = st.button("Generate Blog")


if submit:
    st.write(llama_response(input_text, number_of_words, blog_type))
