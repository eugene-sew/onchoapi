"""
Install the Google AI Python SDK

$ pip install google-generativeai
$ pip install google.ai.generativelanguage
"""

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key="AIzaSyBSX4AmvGJ0IEAQ1J2q2RTwkcV3WaO2kKo")

def asker(number):
    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        properties = {
        "response": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "implication": content.Schema(
                type = content.Type.STRING,
            ),
            "recommendation": content.Schema(
                type = content.Type.STRING,
            ),
            "prediction": content.Schema(
                type = content.Type.STRING,
            ),
            },
        ),
        },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,

    )

    chat_session = model.start_chat(
    history=[ ]
    )

    response = chat_session.send_message(f"I just took a PCA test and had Gleason Score {number}, what does that mean?")

    return (response.text)