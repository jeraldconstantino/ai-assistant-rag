import streamlit as st
from loguru import logger

# Custom libraries
from core.config import settings
from api_requests.inference import make_inference_request
from api_requests.ingestion import ingest_files


class App:

    def __init__(self):
        """Initialize the application."""
        self.TITLE = settings.title
        self.HEIGHT = settings.height
        self.ICON = settings.icon
        self.LAYOUT = settings.layout

        st.set_page_config(page_title=self.TITLE, 
                           page_icon=self.ICON, 
                           layout=self.LAYOUT)
        st.header(self.TITLE)
        self.messages = st.container(height=self.HEIGHT)

    def check_session_state(self):
        """Initialize session state variables."""
        if "conversation" not in st.session_state:
            st.session_state.conversation = [{
                "user": None,
                "assistant": "Hello 👋! I'm your AI Assistant. I can answer your questions based on your uploaded documents. You can also attach files like PDF, DOCX, or TXT to help me assist you better."
            }]

        if "clicked" not in st.session_state:
            st.session_state.clicked = False

    def generate_message(self, 
                         user_input: str, 
                         temperature: float, 
                         max_tokens: int, 
                         top_p: float, 
                         frequency_penalty: float, 
                         presence_penalty: float):
        """
        Simulate assistant response and update conversation.
        
        Args:
            user_input (str): The user's input message.
            temperature (float): Sampling temperature for the LLM.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Frequency penalty for token generation.
            presence_penalty (float): Presence penalty for token generation.
        """
        
        N = 10  # Number of past turns to keep
        recent_conversation = st.session_state.conversation[-N:]

        payload = {
            "ai_model_parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
        }

        # Build chat history text
        history_text = ""
        for turn in recent_conversation:
            if turn["user"]:
                history_text += f"User: {turn['user']}\n"
            if turn["assistant"]:
                history_text += f"Assistant: {turn['assistant']}\n"
                
        # Use lightweight LLM call to detect user intent
        intent_prompt = (
            f"Classify the following user message as either 'conversation' or 'document'. "
            f"Only respond with 'conversation' or 'document', nothing else.\n\n"
            f"User message: {user_input}"
        )

        logger.info(f"Detecting user intent for the message: {user_input}")

        payload["query"] = intent_prompt
        intent = make_inference_request(payload, invoke_type="direct")
        logger.info(f"Detected intent: {intent}")

        # Route based on detected intent
        if intent == "conversation":
            chat_prompt = (
                f"You are a friendly, helpful AI assistant having an ongoing conversation. "
                f"Here is the conversation so far:\n\n"
                f"{history_text}\n"
                f"Now, respond naturally to the user's latest message:\n\n"
                f"{user_input}"
            )

            payload["query"] = chat_prompt
            assistant_response = make_inference_request(payload, invoke_type="direct")

        elif intent == "document":
            logger.info(f"Generating response for document intent.")
            payload["query"] = user_input
            payload["history"] = history_text
            assistant_response = make_inference_request(payload, invoke_type="indirect")

        else:
            assistant_response = (
                "I'm sorry, I couldn't determine the type of your message. "
                "Please rephrase or ask your document-related question again."
            )

        logger.info(f"Generated response for conversation intent: {assistant_response}")
        
        # Update conversation and render latest message pair
        st.session_state.conversation.append({
            "user": user_input,
            "assistant": assistant_response,
        })
        self.messages.chat_message("user").write(user_input)
        self.messages.chat_message("assistant").write(assistant_response)

    def toggle_clicked(self):   
        """Toggle the state of the file uploader."""
        st.session_state.clicked = not st.session_state.clicked

    def start_session(self):
        """Start the Streamlit session and render the UI components."""
        self.check_session_state()

            # Sidebar LLM Settings
        with st.sidebar:
            st.header("LLM Settings")

            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 1000, 100)
            top_p = st.slider("Top P (Nucleus Sampling)", 0.0, 1.0, 1.0, 0.1)
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)

        for entry in st.session_state.conversation:
            if entry["user"]:
                self.messages.chat_message("user").write(entry["user"])
            if entry["assistant"]:
                self.messages.chat_message("assistant").write(entry["assistant"])

        col1, col2 = st.columns([8, 1], gap="small")
        with col1:
            prompt = st.chat_input("Ask me anything...", key="prompt")

        if prompt:
            logger.info(f"The user prompted: {prompt}")
            self.generate_message(user_input=prompt, 
                                  temperature=temperature, 
                                  max_tokens=max_tokens, 
                                  top_p=top_p, 
                                  frequency_penalty=frequency_penalty, 
                                  presence_penalty=presence_penalty)

        # File uploader logic
        with col2:
            if st.session_state.clicked is True:
                st.button("❌ Close Files", on_click=self.toggle_clicked)
            else:
                st.button("📎 Attach Files", on_click=self.toggle_clicked)

        if st.session_state.clicked:
            uploaded_files = st.file_uploader(
                "Upload files", type=["pdf", "txt", "docx"], 
                accept_multiple_files=True, 
                key="file_uploader"
            )

            if uploaded_files:
                logger.info(f"Ingesting the files")
                files = list()
                for file in uploaded_files:

                    if not file.name.endswith((".pdf", ".txt", ".docx")):
                        st.error("Unsupported file type. Please upload a PDF, TXT, or DOCX file.")
                        continue

                    files.append(("files", (file.name, file, file.type)))

                st.success("Files uploaded successfully!")
                logger.info(f"Files uploaded successfully!")
                logger.info(f"Starting file ingestion process")

                with st.spinner("Ingesting files, please wait..."):
                    response = ingest_files(files)

                    if response.status_code == 200:
                        logger.info(f"Files successfully ingested!")
                        st.success("Files successfully ingested!")

                    else:
                        logger.error(f"Failed to ingest files: {response.status_code}")
                        st.error(f"Failed to ingest files: {response.status_code}")