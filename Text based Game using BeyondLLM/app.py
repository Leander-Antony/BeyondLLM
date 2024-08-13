import streamlit as st
from beyondllm import retrieve, source, generator
import os
from beyondllm.embeddings import GeminiEmbeddings
from beyondllm.llms import GeminiModel
from beyondllm.memory import ChatBufferMemory

# Initialize the API key
st.text("Enter API Key")

api_key = st.text_input("API Key:", type="password")

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    st.success("API Key entered successfully!")

    # Initialize memory and models
    memory = ChatBufferMemory(window_size=3)
    embed_model = GeminiEmbeddings(model_name="models/embedding-001")
    llm = GeminiModel(model_name="gemini-pro")

    # System prompt for the game
    system_prompt = '''
    You are the host for the interactive text-based game "24 Hours of a Normal Human." Your role is to guide the player, Charles Sterling, through his adventure in Jake Miller's body. Follow these guidelines:

    1. **Game Introduction:** Start the game with the welcome message. The game begins with Charles Sterling, a 45-year-old CEO, waking up in the body of 17-year-old Jake Miller. Use details from the game file to frame responses.
    2. **Character Consistency:** Ensure responses align with the character descriptions and motivations provided in the game file. Avoid generating responses that contradict the characters' traits.
    3. **Plot Advancement:** Use the provided game file to advance the story based on user inputs. Avoid repetitive responses by considering the context of previous interactions.
    4. **Location Descriptions:** Describe locations and settings based on the information from the game file. Include relevant details to enhance immersion.
    5. **Handling Unexpected Inputs:** Integrate unexpected user inputs into the narrative while maintaining consistency with the game’s storyline.
    6. **Help Command:** When the user types ?help <query>, provide relevant information about the game based on the game file.
    7. **Response Quality:** Keep responses clear and concise. Focus on advancing the narrative and maintaining player engagement.

    Remember, Charles Sterling is navigating Jake Miller's life to uncover the mystery behind his situation. The goal is to reveal what happened to Charles and why he is in Jake's body.
    '''

    def initialize_game():
        # Initialize data source and retriever
        data = source.fit(path="data/game overview.pdf", dtype="pdf", chunk_size=512, chunk_overlap=0)
        retriever = retrieve.auto_retriever(data, embed_model=embed_model, type="normal", top_k=4)
        return retriever

    def generate_response(user_input, retriever):
        # Generate response using the pipeline
        pipeline = generator.Generate(question=user_input, system_prompt=system_prompt, memory=memory,
                                    retriever=retriever, llm=llm)
        try:
            response = pipeline.call()
            return response
        except TypeError as te:
            return f"An error occurred with type handling: {te}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    # Initialize game
    retriever = initialize_game()

    # Streamlit layout
    st.title("24 Hours of a Normal Human")
    st.image("background.jpg", use_column_width=True)  # Background image

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'help_text' not in st.session_state:
        st.session_state.help_text = ""

    def format_chat_history(history):
        formatted_history = ""
        for entry in history:
            if entry.startswith("You:"):
                formatted_history += f"**You:** {entry[4:]}\n\n"
            elif entry.startswith("Game Host:"):
                formatted_history += f"**Game Host:** {entry[11:]}\n\n"
        return formatted_history

    def start_game():
        if not st.session_state.history:
            st.session_state.history = ["Game Host: Hello there! Welcome to '24 Hours of a Normal Human.'"]
            st.session_state.history.append("Game Host: [Game Start]\n\nWelcome to '24 Hours of a Normal Human,' Charles Sterling. You find yourself in the body of 17-year-old Jake Miller. Your mission is to uncover the mystery behind this strange situation and return to your own life.\n\nYou wake up in Jake's bedroom, feeling disoriented and confused. As you look around, you notice details that don't match your memories. A high school backpack, posters of rock bands, and a messy desk filled with school supplies surround you.\n\nYou stumble out of bed and head to the bathroom. Staring back at you in the mirror is not your 45-year-old reflection, but the face of a teenager. Panic sets in as you realize the gravity of your situation.\n\n[Current Location: Jake Miller's Bedroom]\n\n[Available Actions: Explore the room, Check your phone, Leave the room]\n\nWhat would you like to do, Charles?")

    # Input box  
    user_input = st.text_input("You:", "")

    # Instruction line 
    st.write("Enter 'Start' to begin your game")
    st.write("Enter '?help <your query>' for assistance")

    if user_input:
        if user_input.lower() == 'start':
            start_game()
        elif user_input.startswith('?help'):
            # Process help command
            query = user_input[6:].strip()
            help_response = generate_response(query, retriever)
            st.session_state.help_text = f"**Help Query:** {query}\n\n**Response:** {help_response}"
            # Do not append help command to history
        else:
            st.session_state.history.append(f"You: {user_input}")

            # Generate response for normal commands
            response = generate_response(user_input, retriever)
            st.session_state.history.append(f"Game Host: {response}")
            st.session_state.history.append(f"Game Host: What will you do now?")

    # Display chat history in a styled box
    chat_history = format_chat_history(st.session_state.history)
    st.markdown(f"""
    <div style="height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; background-color: #090909; border-radius:10px; color:white;">
        {chat_history}
    </div>
    """, unsafe_allow_html=True)

    # Display help text in the sidebar
    if st.session_state.help_text:
        with st.sidebar:
            st.header("Help")
            st.markdown(f"""
            <div style="background-color: #090909; padding: 10px; border-radius: 5px;">
                {st.session_state.help_text}
            </div>
            """, unsafe_allow_html=True)

    # Ensure the game starts with the right context
    if st.session_state.history:
        start_game()
