from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from src.config import GOOGLE_API_KEY, TEMPERATURE, MAX_TOKENS


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=TEMPERATURE, 
    max_tokens=MAX_TOKENS, 
    api_key=GOOGLE_API_KEY,
)