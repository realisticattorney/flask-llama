from flask import Flask, request, jsonify
import subprocess
import os
from llama_cpp import Llama

app = Flask(__name__)

MODEL_PATH = "./llama.cpp/llama-2-13b-chat.ggmlv3.q4_0.bin"
MODEL_EXEC = "./llama.cpp/main"
# LLAMA_COMMAND = [
#     MODEL_EXEC,
#     "-t", "8",
#     "-ngl", "1",
#     "-m", MODEL_PATH,
#     "--color",
#     "-c", "2048",
#     "--temp", "0.7",
#     "--repeat_penalty", "1.1",
#     "-n", "-1",
# ]
llm = Llama(model_path=MODEL_PATH)


@app.route('/model', methods=['POST'])
def model():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # command = LLAMA_COMMAND + ["-p", f"[INST] {prompt} [/INST]"]
    # result = subprocess.run(command, capture_output=True, text=True)

    # if result.returncode != 0:
        # return jsonify({'error': 'Model execution failed'}), 500

    # return jsonify({'response': result.stdout})
    output = llm(prompt, max_tokens=2048, temperature=0.7, echo=True)
    return jsonify({'response': output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
