import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATASET_NAME ="coeuslearning/hotel_reviews"
PAGE_CONTENT_COLUMN = "review"

CHUNK_SIZE = 500 
CHUNK_OVERLAP = 75

MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"
MODEL_KWARDS ={'device':'cpu'}
ENCODE_KWARDS = {'normalize_embeddings': False} # False

TEMPERATURE = 0.0 
MAX_TOKENS = None

TOP_K=5



