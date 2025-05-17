from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/analyze": {"origins": "http://localhost:3000"}})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({"error": "No ticker selected"})
    
    try:
        result = subprocess.run(['python', 'analysis.py', ticker], capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({"error": result.stderr})

        # Ensure static folder exists
        if not os.path.exists('static'):
            os.makedirs('static')

        # Find all graphs related to the ticker
        graph_paths = [f for f in os.listdir('static') if f.startswith(f'{ticker}_') and f.endswith('.png')]
        graph_urls = [f"http://127.0.0.1:5000/static/{path}" for path in graph_paths]

        if graph_urls:
            return jsonify({"message": f"Analysis completed for {ticker}", "graphUrls": graph_urls})
        else:
            return jsonify({"error": "Graph generation failed or no graphs were found."})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/static/<path:path>')
def serve_static(path):
    try:
        return send_from_directory('static', path)
    except Exception as e:
        return jsonify({"error": f"Error serving file: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)





