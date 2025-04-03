from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
MODEL_PATH = "./Deepseek-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model.eval()  # Set model to evaluation mode
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/complete', methods=['POST'])
def complete():
    try:
        # Ensure request is JSON
        if not request.is_json:
            return jsonify({'error': 'Invalid content type. Expected application/json'}), 400

        data = request.get_json()

        # Validate input
        if not isinstance(data, dict) or 'text' not in data:
            return jsonify({'error': 'Missing or invalid "text" field in request'}), 400

        input_text = data['text'].strip()
        if not input_text:
            return jsonify({'error': 'Empty input text'}), 400

        logger.info(f"Processing completion request for input length: {len(input_text)}")

        # Generate completion
        with torch.no_grad():  # Disable gradient calculation for inference
            inputs = tokenizer(input_text, return_tensors="pt")  # No need to move to device for CPU
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        logger.info(f"Generated completion of length: {len(completion)}")
        return jsonify({'completion': completion})
    
    except Exception as e:
        logger.error(f"Error during completion: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')