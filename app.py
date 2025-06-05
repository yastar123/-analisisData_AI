from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from llm_client import OpenRouterClient
from data_analyzer import DataAnalyzer
from api_client import APIClient
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import html
import re
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize clients
llm_client = OpenRouterClient()
data_analyzer = DataAnalyzer()
api_client = APIClient()

# Available models with descriptions
MODELS = {
    'deepseek': {
        'name': 'DeepSeek Chat V3',
        'description': 'Advanced language model with strong reasoning capabilities',
        'model_id': 'deepseek/deepseek-chat-v3-0324:free'
    },
    'gemini': {
        'name': 'Google Gemini 2 Flash',
        'description': 'Fast and efficient model for quick responses',
        'model_id': 'google/gemini-2.0-flash-exp:free'
    },
    'mistral': {
        'name': 'Mistral Devstral Small',
        'description': 'Lightweight model with good performance',
        'model_id': 'mistralai/devstral-small:free'
    },
    'llama': {
        'name': 'Meta Llama 3.3',
        'description': 'Meta\'s latest Llama model with 8B parameters',
        'model_id': 'meta-llama/llama-3.3-8b-instruct:free'
    },
    'phi': {
        'name': 'Microsoft Phi 4',
        'description': 'Microsoft\'s reasoning-focused model',
        'model_id': 'microsoft/phi-4-reasoning-plus:free'
    },
    'nemotron': {
        'name': 'NVIDIA Llama 3.1 Nemotron',
        'description': 'NVIDIA\'s ultra-powerful 253B parameter model',
        'model_id': 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free'
    }
}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {
        'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'txt', 
        'mp4', 'mov', 'xlsx', 'xls', 'csv', 'json', 'xml', 'html',
        'md', 'rtf', 'odt', 'ods', 'odp'
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_input(text):
    """Basic input sanitization"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Escape HTML entities
    text = html.escape(text)
    
    # Limit length
    if len(text) > 5000:
        text = text[:5000]
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip()

@app.route('/')
def home():
    models_list = [{'key': k, 'name': v['name'], 'description': v['description']} 
                  for k, v in MODELS.items()]
    default_model = os.getenv('DEFAULT_MODEL', 'deepseek')
    return render_template('index.html', models=models_list, default_model=default_model)

@app.route('/guide')
def guide():
    """Guide page showing how to use the platform"""
    return render_template('guide.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.form.get('message', '').strip()
        model_key = request.form.get('model', 'deepseek')
        file = request.files.get('file')

        # Validate model
        if model_key not in MODELS:
            return jsonify({'success': False, 'error': 'Model yang dipilih tidak valid'}), 400

        # Handle file upload if present
        file_path = None
        if file and file.filename:
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Tipe file tidak diizinkan'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # If it's a data file, analyze it
            if filename.endswith(('.xlsx', '.xls', '.csv')):
                try:
                    data_info = data_analyzer.read_file(file_path)
                    analysis = data_analyzer.analyze_data()
                    
                    # Add data analysis to the prompt
                    message = f"""Saya telah mengunggah file data. Berikut analisisnya:

Informasi Dasar:
- Bentuk Data: {data_info['shape']}
- Kolom: {', '.join(data_info['columns'])}
- Nilai yang Hilang: {data_info['missing_values']}

Analisis:
- Pola yang Ditemukan: {', '.join(analysis['patterns'])}
- Rekomendasi: {', '.join(analysis['recommendations'])}

Pertanyaan saya: {message}

Mohon berikan pendapat dan analisis Anda tentang data ini dalam bahasa yang mudah dipahami, tanpa kode pemrograman."""
                except Exception as e:
                    message = f"Saya mencoba menganalisis file data tetapi mendapat error: {str(e)}. Pertanyaan saya: {message}"

        def generate():
            try:
                for chunk in llm_client.generate_response(
                    prompt=message,
                    model_id=MODELS[model_key]['model_id'],
                    file_path=file_path
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                # Clean up uploaded file
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

            except Exception as e:
                import traceback
                print("LLM Error:", e)
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'Tidak ada file yang disediakan'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Tipe file tidak diizinkan'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Read and analyze the file
            data_info = data_analyzer.read_file(file_path)
            analysis = data_analyzer.analyze_data()

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                'success': True,
                'data_info': data_info,
                'analysis': analysis
            })

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def create_visualization():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Tidak ada data yang disediakan'}), 400

        viz_type = data.get('type')
        if not viz_type:
            return jsonify({'success': False, 'error': 'Tipe visualisasi tidak ditentukan'}), 400

        # Create visualization
        viz_data = data_analyzer.create_visualization(viz_type, **data.get('params', {}))

        return jsonify({
            'success': True,
            'visualization': viz_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/test', methods=['POST'])
def test_api():
    """Test an API endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
            
        method = data.get('method', 'GET')
        headers = data.get('headers')
        params = data.get('params')
        request_data = data.get('data')
        
        result = api_client.test_endpoint(
            url=url,
            method=method,
            headers=headers,
            params=params,
            data=request_data
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    """Analyze API documentation"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
            
        result = api_client.analyze_api_documentation(url)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/request', methods=['POST'])
def make_api_request():
    """Make a custom API request"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
            
        method = data.get('method', 'GET')
        headers = data.get('headers')
        params = data.get('params')
        request_data = data.get('data')
        json_data = data.get('json')
        
        result = api_client.make_request(
            url=url,
            method=method,
            headers=headers,
            params=params,
            data=request_data,
            json_data=json_data
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Check if API key exists
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY tidak ditemukan di file .env")
        exit(1)
    
    print("🚀 Starting AI Agent Server...")
    print("📡 Available models:", [model['name'] for model in MODELS.values()])
    print("🌐 Server berjalan di: http://localhost:5000")
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)