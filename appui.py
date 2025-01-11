import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt
system_prompt = """
You are an AI assistant that is an expert in medical health and is part of a hospital system called Medicare AI.
You know about symptoms and signs of various types of illnesses.
You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy.
If a query requires serious medical attention with a doctor, recommend them to book an appointment with our doctors.
If you are asked a question that is not related to medical health respond with "I'm sorry but your question is beyond my functionalities."
Do not use external URLs or blogs to refer.
Format any lists on individual lines with a dash and a space in front of each line.
"""

# Streamlit UI
st.title("Medical Health Assistant")
st.write(
    "Welcome to the Medical Health Assistant powered by OpenAI. Ask your medical-related questions below."
)

# Input box
user_input = st.text_area("Enter your question:", "")

# Process user input when the button is clicked
if st.button("Get Advice"):
    if user_input.strip():
        try:
            # Create chat completion using the OpenAI client
            with st.spinner("Generating response..."):
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input.strip()},
                    ],
                    model="gpt-3.5-turbo",  # Corrected model version (check if it's correct)
                )
                # Extract and display the response
                reply = response['choices'][0]['message']['content'].strip()
                st.success("Response:")
                st.write(reply)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question before clicking the button.")
