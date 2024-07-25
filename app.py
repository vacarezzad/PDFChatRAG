
import streamlit as st

from functions import (DocumentLoader, add_message_to_history,
                       configure_qa_chain, display_chat_history,
                       initialize_chat_history, process_user_query)

st.set_page_config(page_title="Chate con un documento", page_icon="🦜")
st.title("📚 Chatea con un documento")

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()




def main():
    """Main function to initialize and manage the chat interface."""
    # ================================================================
    # Initialize Chat History
    # ================================================================
    initialize_chat_history()

    # ================================================================
    # Clear Chat History Button
    # ================================================================
    if st.button("Clear Chat"):
        st.session_state['chat_history'] = []

    # ================================================================
    # Configure QA Chain
    # ================================================================
    qa_chain = configure_qa_chain(uploaded_files)
    
    # ================================================================
    # User Input
    # ================================================================
    user_query = st.chat_input(placeholder="Ask me anything!")
    
    if user_query:
        # ================================================================
        # Store User Message in Chat History
        # ================================================================
        add_message_to_history("user", user_query)
        
        # ================================================================
        # Process User Query and Store Response
        # ================================================================
        response = process_user_query(qa_chain, user_query)
        add_message_to_history("assistant", response)
    
    # ================================================================
    # Display Chat History
    # ================================================================
    display_chat_history()

# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    main()
