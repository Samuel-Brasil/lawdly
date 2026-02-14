import streamlit as st
import openai
import PyPDF2
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI
import base64

import pandas as pd
from datetime import datetime
import requests

import anthropic
# import google.generativeai as genai

from google import genai
from google.genai import types
import os

import threading
import time
import random  # for random choice

###############################################################################
# Streamlit Page Setup
###############################################################################

st.set_page_config(layout="wide")
with st.sidebar:

    img1a = 'assets/logo.jpeg'
    st.image([img1a], width=300)
#    st.markdown(
#        f'<div><span style="color:#750014; font-size:40px; font-weight:bold;">Lawdly</span>',
#        unsafe_allow_html=True
#    )
    
#    st.write('#### Coder:')
#    img2logo = 'assets/logo.jpeg'
#    st.image([img2logo], caption=["Sam"], width=60)

    
    # Getting the user's email
#    user_email = st.experimental_user.email
    user_email = "test"

    # Model selection
    st.write('#### Model Selection:')
    model_option = st.selectbox(
        'Choose model',
        (
            'gpt-5.2', 'gpt-5.1', 'gpt-5-mini', 
            'gemini-3-pro-preview', 'gemini-3-flash-preview', 
            'claude-opus-4-5', 'claude-sonnet-4-5',
            'deepseek-reasoner', 'deepseek-chat',
            'Agentic_AI-by_Sam'
        )
    )

###############################################################################
# GitHub Access to the case folder and to save generated responses
owner = 'Samuel-Brasil'
repo = 'BenchMarks'
github_token = st.secrets['LAWDLY_GITHUB_TOKEN']

DATA_FOLDER = "data"
GITHUB_API_URL = f'https://api.github.com/repos/{owner}/{repo}/contents/{DATA_FOLDER}'

headers = {
    'Authorization': f'token {github_token}'  
}

response_cases = requests.get(GITHUB_API_URL, headers=headers)

if response_cases.status_code == 200:
    files = response_cases.json()
    pdf_files = {file['name']: file['download_url'] for file in files if file['name'].endswith('.pdf')}
else:
    st.write("⚠️ Failed to fetch files from GitHub. Check your repository access and token.")
    pdf_files = {}

################################################################################
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ''
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    return pdf_text

# Function to select the LLM from the models available
def get_LLM_function(model_option):
    openai_models = ['gpt-5.2', 'gpt-5.1', 'gpt-5-mini']
    google_models = ['gemini-3-pro-preview', 'gemini-3-flash-preview']
    anthropic_models = ['claude-opus-4-5', 'claude-sonnet-4-5']
    deepseek_models = ['deepseek-chat', 'deepseek-reasoner']
    anthropic_new = ['claude-3-7-sonnet-latest']
    sam_models = ['Agentic_AI-by_Sam']

    if model_option in openai_models:
        return LLM_openai
    elif model_option in google_models:
        return LLM_google
    elif model_option in anthropic_models:
        return LLM_anthropic
    elif model_option in deepseek_models:
        return LLM_deepseek
    elif model_option in anthropic_new:
        return LLM_anthropic_new
    elif model_option in sam_models:
        return LLM_Sam
    else:
        return "Work in Progress: {model_option}"

# Function to use OpenAI's models
def LLM_openai(prompt, model_option):
    client = OpenAI()
    MODEL = model_option
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

# Function to use Google's Gemini
def LLM_google(prompt, model_option):
    MODEL = model_option
    client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])
    response = client.models.generate_content(
        model='gemini-3-pro-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        ),
    )
    return response.text

# Function to use Anthropic's Claude
def LLM_anthropic(prompt, model_option):
    MODEL = model_option
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_option,
        max_tokens=1000,
        temperature=0,
        system="You are a world-class Law Professor at PhD level. Respond only with full explanations",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text


# Function to new Anthropic
def LLM_anthropic_new(prompt, model_option):
    MODEL = model_option
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_option,
        max_tokens=1000,
        temperature=0,
        system="You are a world-class Law Professor at PhD level. Respond only with full explanations",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text
    

# Function to use DeepSeek's models
def LLM_deepseek(prompt, model_option):
    client = OpenAI(api_key=st.secrets['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
    MODEL = model_option
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
#        max_tokens=1024,
#        temperature=0.7,
        stream=False
    )
    return completion.choices[0].message.content


# Function to use Agentic_Graph-RAG
def LLM_Sam(prompt, model_option):
    message = 'The Agentic_AI-by_Sam is a multi-agent reasoning model currently in training and is expected to be available within the next week.'
    return message


###############################################################################
# FUNCTION TO SAVE QUERIES TO GITHUB REPO
def saving_queries(date, user_email, model, file_pdf, prompt, response):
    path_to_file = 'queries.xlsx'
    github_token = st.secrets['BENCHMARK_GITHUB_TOKEN']

    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path_to_file}'
    headers = {'Authorization': f'token {github_token}'}
    response_get = requests.get(url, headers=headers)

    if response_get.status_code == 200:
        content = response_get.json()
        sha = content['sha']
        file_content = base64.b64decode(content['content'])
        df = pd.read_excel(BytesIO(file_content))
    else:
        df = pd.DataFrame(columns=['date', 'user_email', 'model', 'prompt', 'response', 'ground_truth', 'evaluation'])
        sha = None

    new_row = {
        'date': date,
        'user_email': user_email,
        'model': model,
        'file_pdf': file_pdf,
        'prompt': prompt,
        'response': response,
        'ground_truth': '',
        'evaluation': ''
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    excel_content = excel_buffer.read()
    b64_content = base64.b64encode(excel_content).decode('utf-8')

    data = {
        'message': 'Update queries.xlsx',
        'content': b64_content,
        'branch': 'main'
    }
    if sha:
        data['sha'] = sha

    requests.put(url, headers=headers, json=data)

###############################################################################
# Generate the LLM response in a separate thread and show rotating messages
def generate_response(prompt, model_option):
    """
    1. Start a timer.
    2. Launch the LLM call in a separate thread.
    3. Show rotating messages while the thread runs.
    4. Once the LLM is done, remove rotating messages and display total time.
    """

    # Record start time
    start_time = time.time()
    
    # We'll store the LLM response in this dict so we can retrieve it
    response_holder = {}

    # Define the worker thread function
    def worker():
        LLM_function = get_LLM_function(model_option)
        resp = LLM_function(prompt, model_option)
        response_holder['response'] = resp

    # Start the worker thread
    t = threading.Thread(target=worker)
    t.start()

    # Prepare a placeholder for the rotating messages
    placeholder = st.empty()

    # List of rotating messages
    messages = [
        "Analyzing your request...",
        "Reading the Supreme Court opinion carefully...", 
        "Hold on...",
        "Crunching the data...",
        "One moment...",
        "Taking a closer look at the text...",
        "Generating the best response...",
        "Trying to understand the Supreme Court...",
        "It's not that easy...",
        "Hang tight..."
    ]

    i = 0
    # Keep cycling messages while thread is alive
    while t.is_alive():
        placeholder.info(messages[i % len(messages)])
        time.sleep(2)
        i += 1

    # Thread finished; ensure it's fully done
    t.join()
    
    # Remove the rotating messages
    placeholder.empty()

    # Compute total elapsed time
    total_time = int(time.time() - start_time)

    # Return the response plus the total time
    return response_holder['response'], total_time

###############################################################################
def main():
    teste='testando a birosca'
    print(teste)
    st.markdown(
        f'<div><span style="color:#750014; font-size:55px; font-weight:bold;">Lawdly</span>',
#        f'<span style="color:#8b959e; font-size:55px; font-weight:bold;">Mark</span></div>',
        unsafe_allow_html=True
    )

    
# NEW CODE TO SELECT PDF FROM GITHUB FOLDER
    # **PDF Upload with Selectbox**
    st.write('#### Select a Supreme Court Opinion:')

    if pdf_files:
        selected_file_key = st.selectbox(
            'Choose Case:',
            list(pdf_files.keys())
        )

        if selected_file_key:
            file_url = pdf_files[selected_file_key]

            # Download the selected PDF file
            response = requests.get(file_url, headers=headers)

            if response.status_code == 200:
                pdf_data = BytesIO(response.content)
                text = extract_text_from_pdf(pdf_data)
            
                st.write(f"✅ **Text extracted from:** `{selected_file_key}`")
                st.write("✅ Ready to proceed.")
            else:
                st.write("⚠️ Failed to download the selected file.")
    else:
        st.write("⚠️ No PDF files found in the specified GitHub folder.")


    query_examples = [
        ("Case summarization", "Make a complete summarization of the provided document"),
        ("Slippery Slope - Prompt 1", 
         "Quote all slippery slope arguments you find in the provided text"),
        ("Slippery Slope - Prompt 2", 
         "Quote all slippery slope arguments you find in the provided text, explaning why it is a slippery slope"),
        ("Check Justices' Sympathy", 
#         "Check if all Justices have sympathy for a party or group, transcribe in quotation marks the text containing the part that shows sympathy and explain why."),
          "Quote all slippery slope arguments you find in the provided text, explaning if it indicates a sympathy for a party or group"),
    ]

    num_buttons_per_row = 4
    for idx in range(0, len(query_examples), num_buttons_per_row):
        cols = st.columns(num_buttons_per_row)
        for col, (button_label, query) in zip(cols, query_examples[idx:idx + num_buttons_per_row]):
            if col.button(button_label):
                # Build the full prompt
                if pdf_files is not None:
                    LLM_full_prompt = f"Content from PDF:\n{text}\n\nUser's question: {query}\nAnswer:"
                else:
                    LLM_full_prompt = f"User's question: {query}\n(No PDF file uploaded.)"

                # Generate the response, returning (answer, total_time)
                answer, total_time = generate_response(LLM_full_prompt, model_option)

                # Show total elapsed time
                st.success(f"Time elapsed: {total_time} seconds")

                # Display the final response
                st.write(answer)

                # Save to GitHub
                date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                saving_queries(date_str, user_email, model_option, selected_file_key if pdf_files else "",
                               query, answer)


if __name__ == "__main__":
    main()

