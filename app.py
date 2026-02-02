"""
Flask Application - RAG com Múltiplos Embedding Models
"""

import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify
)
from werkzeug.utils import secure_filename
from rag_engine import rag_engine

# Configuração do Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Criar pasta de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensões permitidas
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    """Verifica se o arquivo é permitido."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# ROTAS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Página inicial - redireciona para setup ou chat."""
    if session.get('api_configured'):
        if session.get('pdf_loaded'):
            return redirect(url_for('chat'))
        return redirect(url_for('upload'))
    return redirect(url_for('setup'))


@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Página de configuração de API keys."""
    if request.method == 'POST':
        openai_key = request.form.get('openai_key', '').strip()
        llama_key = request.form.get('llama_key', '').strip()
        jina_key = request.form.get('jina_key', '').strip()
        
        # Validar campos obrigatórios
        if not openai_key or not llama_key:
            flash('OpenAI e LlamaCloud API keys são obrigatórias!', 'error')
            return render_template('setup.html')
        
        # Configurar API keys
        try:
            rag_engine.set_api_keys(
                openai_key=openai_key,
                llama_cloud_key=llama_key,
                jina_key=jina_key if jina_key else None
            )
            
            # Salvar na sessão
            session['api_configured'] = True
            session['has_jina'] = bool(jina_key)
            
            flash('API keys configuradas com sucesso!', 'success')
            return redirect(url_for('upload'))
            
        except Exception as e:
            flash(f'Erro ao configurar API keys: {str(e)}', 'error')
            return render_template('setup.html')
    
    return render_template('setup.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Página de upload de PDF."""
    if not session.get('api_configured'):
        flash('Configure as API keys primeiro.', 'warning')
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Verificar se arquivo foi enviado
        if 'pdf_file' not in request.files:
            flash('Nenhum arquivo selecionado.', 'error')
            return render_template('upload.html')
        
        file = request.files['pdf_file']
        
        if file.filename == '':
            flash('Nenhum arquivo selecionado.', 'error')
            return render_template('upload.html')
        
        if file and allowed_file(file.filename):
            # Salvar arquivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Obter modelos selecionados
            selected_models = request.form.getlist('models')
            if not selected_models:
                selected_models = ['openai', 'bge']
            
            try:
                # Processar PDF
                flash('Processando PDF... Isso pode levar alguns minutos.', 'info')
                
                result = rag_engine.load_pdf(filepath)
                
                if not result['success']:
                    flash(f'Erro ao processar PDF: {result.get("error")}', 'error')
                    return render_template('upload.html')
                
                # Inicializar embedding models
                rag_engine.initialize_embedding_models(selected_models)
                
                # Criar índices
                index_results = rag_engine.create_indexes()
                
                # Verificar resultados
                successful = [k for k, v in index_results.items() if v]
                
                if successful:
                    session['pdf_loaded'] = True
                    session['pdf_filename'] = filename
                    session['active_models'] = successful
                    
                    flash(f'PDF processado! {result["num_chunks"]} chunks criados. '
                          f'Modelos ativos: {", ".join(successful)}', 'success')
                    return redirect(url_for('chat'))
                else:
                    flash('Erro ao criar índices.', 'error')
                    
            except Exception as e:
                flash(f'Erro: {str(e)}', 'error')
        else:
            flash('Tipo de arquivo não permitido. Use PDF.', 'error')
    
    return render_template('upload.html', has_jina=session.get('has_jina', False))


@app.route('/chat')
def chat():
    """Página de chat/consulta."""
    if not session.get('api_configured'):
        return redirect(url_for('setup'))
    
    if not session.get('pdf_loaded'):
        flash('Faça upload de um PDF primeiro.', 'warning')
        return redirect(url_for('upload'))
    
    return render_template(
        'chat.html',
        pdf_filename=session.get('pdf_filename', 'Documento'),
        active_models=session.get('active_models', []),
        status=rag_engine.get_status()
    )


@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint para queries."""
    if not session.get('pdf_loaded'):
        return jsonify({'error': 'Nenhum documento carregado'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    model = data.get('model', 'all')
    
    if not question:
        return jsonify({'error': 'Pergunta não pode estar vazia'}), 400
    
    try:
        results = rag_engine.query(question, model)
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'model': r.model_name,
                    'response': r.response,
                    'sources': r.sources
                }
                for r in results
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Retorna o status do sistema."""
    return jsonify(rag_engine.get_status())


@app.route('/reset')
def reset():
    """Reseta a sessão."""
    session.clear()
    flash('Sessão resetada.', 'info')
    return redirect(url_for('setup'))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)