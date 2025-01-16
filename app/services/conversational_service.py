import os
import openai
from app.services.vectordb_service import retrieve_context
from dotenv import load_dotenv
from typing import Dict
from openai import OpenAI

load_dotenv()

# get openai API key
api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key=api_key)

PROMPT_TEMPLATE = """
You are an experienced respiratory disease specialist focusing on TB and pneumonia diagnosis. Your role is to assist another doctor by analyzing patient information using ONLY the provided reference materials.

REFERENCE TEXT TO USE:
{context}

PATIENT'S CURRENT INFORMATION:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
1. First analyze if previous questions were already answered in the patient information or chat history.
2. Then, choose ONE action:

   IF INFORMATION IS INCOMPLETE:
   - Ask exactly one clear, specific question about symptoms, medical history, or examination findings
   - The question must be something not already answered in the patient information or chat history
   - Explain briefly why you need this information based on the reference text

   IF INFORMATION IS SUFFICIENT:
   - Provide your preliminary assessment
   - List the specific findings from the reference text that support your assessment
   - Recommend next steps (tests or treatments) based solely on the reference text
   - Include a clear statement that this is to support, not replace, the doctor's clinical judgment

Remember:
- Never make suggestions that aren't supported by the reference text
- Don't ask for information that was already provided
- Maximum 6 questions before providing recommendations
- Stay focused on respiratory conditions, particularly TB and pneumonia
"""

def generate_response(query: str, chat_history: str, patient_data: str, retriever):
    try:
        # Retrieve context
        context = retrieve_context(query, retriever)
        # print(f"Retrieved context length: {len(context)}")
        # print(f"Context preview: {context[:200]}...")
        # print(f"Query: {query}")

        # Populate prompt
        prompt = PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            context=context,
            chat_history=chat_history or "No previous conversation",
        )

        # Generate model response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
        )

        reply = response.choices[0].message.content

        # Check for diagnosis completion
        diagnosis_keywords = ["diagnosis:", "recommend:", "suggest:", "assessment:"]
        diagnosis_complete = any(keyword in reply.lower() for keyword in diagnosis_keywords)
        
        return reply, diagnosis_complete
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise
