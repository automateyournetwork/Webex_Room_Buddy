import os
import urllib3
import requests
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings 

urllib3.disable_warnings()

@st.cache_resource
def load_model():
    embedding_model=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
    return embedding_model

class ChatWithWebexRoom:
    def __init__(self, key, room, file_path='conversation.txt'):
        self.embedding_model = load_model()
        self.conversation_history = []
        self.fetch_and_setup_conversation(key, room)
        self.load_text_txt(file_path)
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def fetch_and_setup_conversation(self, key, room):
        with st.spinner("Gathering Webex Room Messages ...."):
            url = f"https://webexapis.com/v1/messages?roomId={room}&max=500"
            headers = {'Authorization': f'Bearer {key}'}
            response = requests.get(url, headers=headers, verify=False)
            if response.status_code == 200:
                st.write("Webex status code", response.status_code)
                json_messages = response.json()
                st.write("Raw JSON", json_messages)
                conversation = ""
                for message in json_messages['items']:
                    if 'text' in message:
                        person = message['personEmail'].split("@")[0]  # Simplified email handling
                        timestamp = message['created']
                        conversation += f"{timestamp} {person}: {message['text']}\n"
                self.save_conversation_to_file(conversation, 'conversation.txt')
            else:
                st.error("Failed to fetch messages from Webex.")
                self.pages = []
        
    def save_conversation_to_file(self, conversation, file_name):
        with open(file_name, 'w') as file:
            file.write(conversation)

    def load_text_txt(self, file_path):
        with st.spinner("Loading Text.."):
            # Load and process the TXT file
            self.loader = TextLoader(file_path=file_path)
            self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        with st.spinner("Chunking ... "):
            # Create a text splitter
            self.text_splitter = SemanticChunker(self.embedding_model)
            self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        with st.spinner("Storing in chroma..."):
            # Check if vectordb exists and use the .delete() function to clear it
            if hasattr(self, 'vectordb') and self.vectordb is not None:
                # Use the .delete() function to clear or delete the existing vectordb
                self.vectordb.delete()
                # After deletion, you can optionally set vectordb to None or leave it as is
                # since you'll be creating a new instance immediately after
                self.vectordb = None
    
            embeddings = self.embedding_model
            # Continue creating a new vectordb instance as before
            self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
            self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        with st.spinner("Starting Conversational Retrieval Chain"):
            llm = Ollama(model=st.session_state['selected_model'], base_url="http://ollama:11434")
            self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 25}), memory=self.memory)

    def chat(self, question):
        # Format the user's prompt and add it to the conversation history
        user_prompt = f"User: {question}"
        self.conversation_history.append({"text": user_prompt, "sender": "user"})

        # Format the entire conversation history for context, excluding the current prompt
        conversation_context = self.format_conversation_history(include_current=False)

        # Concatenate the current question with conversation context
        combined_input = f"Context: {conversation_context}\nQuestion: {question}"

        # Generate a response using the ConversationalRetrievalChain
        response = self.qa.invoke(combined_input)

        # Extract the answer from the response
        answer = response.get('answer', 'No answer found.')

        # Format the AI's response
        ai_response = f"Webex Room: {answer}"
        self.conversation_history.append({"text": ai_response, "sender": "bot"})

        # Update the Streamlit session state by appending new history with both user prompt and AI response
        st.session_state['conversation_history'] += f"\n{user_prompt}\n{ai_response}"

        # Return the formatted AI response for immediate display
        return ai_response

    def format_conversation_history(self, include_current=True):
        formatted_history = ""
        history_to_format = self.conversation_history[:-1] if not include_current else self.conversation_history
        for msg in history_to_format:
            speaker = "You: " if msg["sender"] == "user" else "Bot: "
            formatted_history += f"{speaker}{msg['text']}\n"
        return formatted_history

def get_ollama_models(base_url):
    try:       
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()
        
        # Extract just the model names for the dropdown
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []

def page_webex_room():
    st.subheader("Pick Your Model")
    # Model selection dropdown
    try:
        models = get_ollama_models("http://ollama:11434/")
        if models:
            selected_model = st.selectbox("Select Model", models)
            st.session_state['selected_model'] = selected_model
        else:
            st.markdown('No models available. Please visit [localhost:3002](http://localhost:3002) to download models.')
    except Exception as e:
        st.error(f"Failed to fetch models: {str(e)}")
    st.subheader("Enter Webex Room Details")
    room = st.text_input("Webex Room ID:", "")
    key = st.text_input("Webex API Key:", type="password")
    if st.button("Fetch Conversation and Proceed"):
        if room and key:
            st.session_state['webex_details'] = {'room': room, 'key': key}
            st.session_state['page'] = 'chat'
        else:
            st.error("Please enter both Webex Room ID and API Key.")

def page_chat():
    user_input = st.text_input("Ask a question about the WebEx Room Messages:", key="user_input")
    if st.button("Ask"):
        if 'selected_model' in st.session_state and st.session_state['selected_model']:
            # Ensure 'webex_details' contains the necessary room and key, set previously
            webex_details = st.session_state['webex_details']
            
            # Check if chat_instance already exists in the session state, if not, create it
            if 'chat_instance' not in st.session_state:
                st.session_state['chat_instance'] = ChatWithWebexRoom(webex_details['key'], webex_details['room'], 'conversation.txt')
            
            chat_instance = st.session_state['chat_instance']
            # Now `chat_instance` is ready with the conversation loaded from the text file
            
            ai_response = chat_instance.chat(user_input)
            st.session_state['conversation_history'] += f"\nUser: {user_input}\nAI: {ai_response}"
            st.text_area("Conversation History:", value=st.session_state['conversation_history'], height=300, key="conversation_history_display")
            # Update and display the conversation history as before
        else:
            st.error("Please select a model to proceed.")
    
    if st.button("Talk to a new Webex Room"):
        # Reset conversation history and any other desired session state keys
        st.session_state['conversation_history'] = ""
        # Optionally, clear the chat_instance to force creating a new one for a new room
        if 'chat_instance' in st.session_state:
            del st.session_state['chat_instance']
        # Reset other states as needed
        st.session_state['page'] = 'webex_room'  # Redirect back to the first page
        st.rerun()
       

# Streamlit UI setup
st.title("Webex Room Buddy")

# Initialize or reset session state keys as needed
if 'page' not in st.session_state:
    st.session_state['page'] = 'webex_room'
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

# Page routing
if st.session_state['page'] == 'webex_room':
    page_webex_room()
elif st.session_state['page'] == 'chat' and 'webex_details' in st.session_state:
    page_chat()