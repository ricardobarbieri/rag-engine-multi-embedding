#!/usr/bin/env python3
"""
Script para iniciar a aplicação Flask RAG
"""

import os
import sys

def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    required = [
        'flask',
        'llama_index',
        'llama_parse',
        'chromadb',
        'nest_asyncio'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Dependências faltando:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstale com: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ Todas as dependências estão instaladas")

def create_directories():
    """Cria diretórios necessários."""
    directories = ['uploads', 'static', 'templates']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Diretórios criados")

def main():
    """Função principal."""
    print("=" * 50)
    print("🚀 RAG Flask Application")
    print("=" * 50)
    
    # Verificar dependências
    check_dependencies()
    
    # Criar diretórios
    create_directories()
    
    # Importar e executar app
    from app import app
    
    print("\n" + "=" * 50)
    print("🌐 Iniciando servidor...")
    print("=" * 50)
    print("\n📍 Acesse: http://localhost:5000")
    print("📍 Ou: http://127.0.0.1:5000")
    print("\n⚠️  Pressione CTRL+C para encerrar\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=True
    )

if __name__ == '__main__':
    main()