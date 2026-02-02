"""
RAG Engine - Motor de Retrieval-Augmented Generation
Suporta múltiplos embedding models: OpenAI, BGE, Jina
"""

import os
import nest_asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter

# LlamaParse para PDFs
from llama_parse import LlamaParse

# Embedding Models
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding

# LLM
from llama_index.llms.openai import OpenAI

# Aplicar nest_asyncio
nest_asyncio.apply()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Resultado de uma query."""
    model_name: str
    response: str
    sources: List[Dict[str, Any]]
    score: float = 0.0


class RAGEngine:
    """
    Motor RAG com suporte a múltiplos embedding models.
    """
    
    def __init__(self):
        self.documents = None
        self.nodes = None
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.query_engines: Dict[str, Any] = {}
        self.embed_models: Dict[str, Any] = {}
        self.is_initialized = False
        self.api_keys = {}
        
        # Configurações de chunking
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Node parser
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def set_api_keys(
        self,
        openai_key: str,
        llama_cloud_key: str,
        jina_key: Optional[str] = None
    ):
        """Configura as API keys."""
        self.api_keys = {
            "openai": openai_key,
            "llama_cloud": llama_cloud_key,
            "jina": jina_key
        }
        
        # Configurar variáveis de ambiente
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key
        if jina_key:
            os.environ["JINA_API_KEY"] = jina_key
        
        # Configurar LLM
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1024,
            api_key=openai_key
        )
        
        logger.info("API keys configuradas com sucesso")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Valida se as API keys estão funcionando."""
        results = {}
        
        # Validar OpenAI
        try:
            embed = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=self.api_keys.get("openai")
            )
            _ = embed.get_text_embedding("test")
            results["openai"] = True
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            results["openai"] = False
        
        # Validar Jina (opcional)
        if self.api_keys.get("jina"):
            try:
                embed = JinaEmbedding(
                    api_key=self.api_keys.get("jina"),
                    model="jina-embeddings-v2-base-en"
                )
                _ = embed.get_text_embedding("test")
                results["jina"] = True
            except Exception as e:
                logger.error(f"Jina validation failed: {e}")
                results["jina"] = False
        
        return results
    
    def load_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Carrega e processa um PDF usando LlamaParse.
        """
        logger.info(f"Carregando PDF: {pdf_path}")
        
        try:
            # Configurar LlamaParse
            parser = LlamaParse(
                api_key=self.api_keys.get("llama_cloud"),
                result_type="markdown",
                verbose=True,
                language="pt"
            )
            
            # Fazer parsing
            self.documents = parser.load_data(pdf_path)
            
            # Criar nodes
            self.nodes = self.node_parser.get_nodes_from_documents(self.documents)
            
            result = {
                "success": True,
                "num_documents": len(self.documents),
                "num_chunks": len(self.nodes),
                "preview": self.documents[0].text[:500] if self.documents else ""
            }
            
            logger.info(f"PDF processado: {result['num_chunks']} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao carregar PDF: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def initialize_embedding_models(self, models: List[str] = None):
        """
        Inicializa os embedding models selecionados.
        """
        if models is None:
            models = ["openai", "bge"]
            if self.api_keys.get("jina"):
                models.append("jina")
        
        self.embed_models = {}
        
        for model in models:
            try:
                if model == "openai":
                    self.embed_models["openai"] = OpenAIEmbedding(
                        model="text-embedding-3-small",
                        api_key=self.api_keys.get("openai")
                    )
                    logger.info("OpenAI Embedding inicializado")
                    
                elif model == "bge":
                    self.embed_models["bge"] = HuggingFaceEmbedding(
                        model_name="BAAI/bge-small-en-v1.5"
                    )
                    logger.info("BGE Embedding inicializado")
                    
                elif model == "jina" and self.api_keys.get("jina"):
                    self.embed_models["jina"] = JinaEmbedding(
                        api_key=self.api_keys.get("jina"),
                        model="jina-embeddings-v2-base-en"
                    )
                    logger.info("Jina Embedding inicializado")
                    
            except Exception as e:
                logger.error(f"Erro ao inicializar {model}: {e}")
        
        return list(self.embed_models.keys())
    
    def create_indexes(self) -> Dict[str, bool]:
        """
        Cria índices vetoriais para cada embedding model.
        """
        if not self.documents:
            raise ValueError("Nenhum documento carregado. Execute load_pdf primeiro.")
        
        results = {}
        
        for name, embed_model in self.embed_models.items():
            try:
                logger.info(f"Criando índice para: {name}")
                
                Settings.embed_model = embed_model
                
                index = VectorStoreIndex.from_documents(
                    self.documents,
                    transformations=[self.node_parser],
                    show_progress=True
                )
                
                self.indexes[name] = index
                self.query_engines[name] = index.as_query_engine(
                    similarity_top_k=3
                )
                
                results[name] = True
                logger.info(f"Índice {name} criado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao criar índice {name}: {e}")
                results[name] = False
        
        self.is_initialized = True
        return results
    
    def query(
        self,
        question: str,
        model: str = "all"
    ) -> List[QueryResult]:
        """
        Executa uma query nos índices.
        
        Args:
            question: Pergunta do usuário
            model: "all" para todos os modelos, ou nome específico
        
        Returns:
            Lista de QueryResult
        """
        if not self.is_initialized:
            raise ValueError("Sistema não inicializado. Execute create_indexes primeiro.")
        
        results = []
        
        # Determinar quais modelos usar
        if model == "all":
            models_to_query = list(self.query_engines.keys())
        else:
            models_to_query = [model] if model in self.query_engines else []
        
        for model_name in models_to_query:
            try:
                # Configurar embed model correto
                Settings.embed_model = self.embed_models[model_name]
                
                # Executar query
                engine = self.query_engines[model_name]
                response = engine.query(question)
                
                # Extrair fontes
                sources = []
                for node in response.source_nodes:
                    sources.append({
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(node.score) if node.score else 0.0,
                        "metadata": dict(node.metadata) if node.metadata else {}
                    })
                
                results.append(QueryResult(
                    model_name=model_name,
                    response=str(response),
                    sources=sources
                ))
                
            except Exception as e:
                logger.error(f"Erro na query com {model_name}: {e}")
                results.append(QueryResult(
                    model_name=model_name,
                    response=f"Erro: {str(e)}",
                    sources=[]
                ))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna o status atual do sistema."""
        return {
            "is_initialized": self.is_initialized,
            "has_documents": self.documents is not None,
            "num_documents": len(self.documents) if self.documents else 0,
            "num_chunks": len(self.nodes) if self.nodes else 0,
            "available_models": list(self.embed_models.keys()),
            "indexes_created": list(self.indexes.keys())
        }


# Instância global do RAG Engine
rag_engine = RAGEngine()