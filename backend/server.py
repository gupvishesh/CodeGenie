from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os
import re
import json
import pyngrok.ngrok as ngrok

# Setup Flask and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "2vDDsjBFnwE7d6orXuyZHEN3toS_2fo2ae8eR5qNJsPAJtgfi")
PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = os.environ.get("MODEL_PATH", "./Deepseek-Instruct")

# Global variable for public URL
public_url = None

# Model configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and tokenizer with proper error handling
try:
    logger.info(f"Loading tokenizer and model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    # We'll handle this in a more graceful way instead of raising
    model = None
    tokenizer = None

