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

# Helper function to ensure model is loaded
def ensure_model():
    if model is None or tokenizer is None:
        return False, {"error": "Model not properly initialized"}, 503
    return True, None, None

# Standardized error response function
def error_response(message, status_code=400):
    logger.error(message)
    return jsonify({"error": message}), status_code

@app.route("/connection_info", methods=['GET'])
def connection_info():
    return jsonify({
        "status": "connected",
        "server_url": public_url,
        "model_status": "ready" if model is not None and tokenizer is not None else "not_ready",
        "device": device
    })

# Endpoint 1: /generate (uses 'prompt')
chat_history=[]

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Check model availability
        model_ready, error_msg, status_code = ensure_model()
        if not model_ready:
            return error_msg, status_code
            
        logger.info("Received request at /generate")
        
        if not request.is_json:
            return error_response("Invalid content type. Expected application/json")
        
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return error_response("No prompt provided")

        logger.info(f"Processing new prompt of length: {len(prompt)}")

        # Format context from history
        context = ""
        for i, (prev_prompt, prev_response) in enumerate(chat_history):
            context += f"---\nPrompt {i+1}: {prev_prompt}\nResponse {i+1}: {prev_response}\n"
        

        full_prompt = f"Prompt: {prompt}\nGive me the response to the mentioned prompt and consider all the below prompts and responses(which are our previous interactions) as context to refer\n {context}"

        logger.info(f"{full_prompt}")

        # Tokenize and generate
        with torch.no_grad():
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            output_tokens = model.generate(
                **inputs,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )

            response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

            # Extract the latest response by removing the context
            if response_text.startswith(full_prompt):
                response_text = response_text[len(full_prompt):].strip()

        logger.info(f"Generated response of length: {len(response_text)}")

        # Save this pair to history
        chat_history.append((prompt, response_text))

        return jsonify({"response": response_text})

    except Exception as e:
        return error_response(f"An internal error occurred: {str(e)}", 500)

# Endpoint 2: /complete (uses 'text')
@app.route('/complete', methods=['POST'])
def complete():
    try:
        # Check model availability
        model_ready, error_msg, status_code = ensure_model()
        if not model_ready:
            return error_msg, status_code
            
        # Validate request format
        if not request.is_json:
            return error_response("Invalid content type. Expected application/json")

        data = request.get_json()
        if not isinstance(data, dict) or 'text' not in data:
            return error_response('Missing or invalid "text" field in request')

        input_text = data['text'].strip()
        if not input_text:
            return error_response('Empty input text')

        logger.info(f"Processing /complete request for input length: {len(input_text)}")

        # Generate completion
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
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
        return error_response(f"Error in /complete: {str(e)}", 500)
    
# Endpoint 3: /hf-complete (uses 'code')
@app.route('/hf-complete', methods=['POST'])
def hf_complete():
    try:
        # Check model availability
        model_ready, error_msg, status_code = ensure_model()
        if not model_ready:
            return error_msg, status_code
            
        # Validate request format
        if not request.is_json:
            return error_response("Invalid content type. Expected application/json")
            
        data = request.get_json()
        input_text = data.get("code", "").strip()

        if not input_text:
            return error_response("No code provided")

        logger.info(f"Received /hf-complete request for input length {len(input_text)}")

        # Generate suggestion
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=32,
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove echoed input
            if full_output.startswith(input_text):
                suggestion = full_output[len(input_text):]
            else:
                suggestion = full_output

            # Stop suggestion at the start of any new function
            lines = suggestion.split('\n')
            filtered = []
            for line in lines:
                if line.strip().startswith("def ") and filtered:
                    break  # Stop if another function is generated
                filtered.append(line)
    
            suggestion = "\n".join(filtered).strip()

        logger.info(f"HF-complete suggestion of length: {len(suggestion)}")
        return jsonify({"completion": suggestion})

    except Exception as e:
        return error_response(f"Error in /hf-complete: {str(e)}", 500)

# Endpoint 4: /fill_in_the_middle (uses 'text')
@app.route("/fill_in_the_middle", methods=["POST"])
def fill_in_the_middle():
    try:
        # Check model availability
        model_ready, error_msg, status_code = ensure_model()
        if not model_ready:
            return error_msg, status_code
            
        # Validate request format
        if not request.is_json:
            return error_response("Invalid content type. Expected application/json")
            
        data = request.get_json()
        if not isinstance(data, dict) or 'text' not in data:
            return error_response('Missing or invalid "text" field in request')
            
        input_text = data['text'].strip()
        if not input_text:
            return error_response('Empty input text')
            
        # Create a better prompt for the fill-in-the-middle task
        prompt = (
            f"{input_text}\n\n"
            "----Complete the code by filling in any gaps or implementing missing functionality.\n"
            "Focus only on providing the missing parts that would make this code more complete.\n"
            "Do not give any other extra block of code, just the the present code which contains the filled missing code "
        )
        
        logger.info(f"Processing /fill_in_the_middle request for input length: {len(input_text)}")

        # Generate completion
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Check if we just got back something too similar to the input
            if completion.strip() == input_text.strip():
                completion = "// No additional code needed"

        logger.info(f"Generated fill-in-the-middle completion of length: {len(completion)}")
        return jsonify({'completion': completion})

    except Exception as e:
        return error_response(f"Error in /fill_in_the_middle: {str(e)}", 500)


# Endpoint 5: /debug (code debugging and analysis)
@app.route('/debug', methods=['POST'])
def debug_code():
    def analyze_code_issues(code):
        # Basic dummy issue detection for placeholder
        issues = []
        if "==" in code and "if" in code:
            issues.append("Potential logical comparison in condition")
        if "print" not in code:
            issues.append("Missing print statement (for debugging output)")
        return issues
    
    try:
        logger.info("Received request at /debug")
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        is_debug_request = "debug" in prompt.lower() or "fix" in prompt.lower() or "issue" in prompt.lower()
        code_to_debug = ""

        if is_debug_request:
            code_match = re.search(r'(?:```[\w]*\n)?(.*?)(?:```)?$', prompt, re.DOTALL)
            if code_match:
                code_to_debug = code_match.group(1).strip()
                logger.info(f"Extracted code for debugging: {code_to_debug[:100]}...")

        if is_debug_request and code_to_debug:
            detected_issues = analyze_code_issues(code_to_debug)
            enhanced_prompt = (
                "Analyze and debug the following code. "
                f"Potential issues detected: {', '.join(detected_issues) if detected_issues else 'None detected'}. "
                "Provide:\n"
                "1. Detailed analysis of problems\n"
                "2. Specific fixes\n"
                "3. Improved version\n"
                "4. Explanation of changes\n"
                f"Code:\n```\n{code_to_debug}\n```"
            )
            prompt = enhanced_prompt
            logger.info(f"Enhanced debug prompt: {enhanced_prompt[:200]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_tokens = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )

        response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

        if is_debug_request:
            if "1." not in response_text and "2." not in response_text:
                response_text = (
                    "## Code Analysis\n\n" +
                    response_text +
                    "\n\n## Suggested Fixes\n\n" +
                    "Here are the recommended changes to improve the code..."
                )

        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()

        logger.info(f"Generated response for /debug (length: {len(response_text)})")
        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error in /debug: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid content type. Expected application/json'}), 400

        data = request.get_json()
        input_text = data.get("text", "").strip()
        if not input_text:
            return jsonify({'error': 'Empty input text'}), 400

        logger.info(f"Processing optimization request with input length: {len(input_text)}")

        # Prepare prompt for the model to generate optimized code
        prompt = (
            "Optimize the following code to improve its time and space complexity. "
            "Use appropriate data structures and algorithms while ensuring the functionality remains the same.\n"
            "Do not include any test cases, only provide the optimized code.\n\n"
            f"{input_text}\n\n# Optimized Code:\n"
        )
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if "# Optimized Code:" in completion:
                code_block = completion.split("# Optimized Code:")[1].strip()
                code_lines = code_block.splitlines()
                func_lines = []
                for line in code_lines:
                    if line.strip().startswith("print(") or line.strip().startswith("# Test Cases"):
                        break
                    func_lines.append(line)
                completion = "\n".join(func_lines).strip()

        logger.info(f"Generated optimized code of length: {len(completion)}")
        return jsonify({'completion': completion})

    except Exception as e:
        logger.exception("Error during optimization")
        return jsonify({'error': str(e)}), 500


# Add health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    model_status = "ready" if model is not None and tokenizer is not None else "not_ready"
    return jsonify({
        "status": "ok",
        "server_url": public_url,
        "model": model_status,
        "device": device
    })

def start_ngrok():
    """
    Start an ngrok tunnel and return the public URL
    """
    global public_url
    
    if not NGROK_AUTH_TOKEN:
        logger.warning("No ngrok auth token provided. Tunnel may have connection limits.")
    else:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
    try:
        # Start ngrok tunnel
        public_url = ngrok.connect(PORT).public_url
        logger.info(f"ngrok tunnel established at: {public_url}")
        
        # Save configuration to a file for clients to access
        with open("server_config.json", "w") as f:
            json.dump({"server_url": public_url}, f)

        logger.info(f"Server URL saved to server_config.json")
        return public_url
    except Exception as e:
        logger.error(f"Error starting ngrok: {str(e)}")
        return None

# Run the flask app
if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = start_ngrok()
    
    if not public_url:
        logger.warning("Failed to established ngrok tunnel. Running with local access only.")
        public_url = f"http://127.0.0.1:{PORT}"

    logger.info(f"Starting Flask server with ngrok tunnel at {public_url}")
    app.run(debug=False, port=PORT, host='0.0.0.0')  # Set debug=False for production