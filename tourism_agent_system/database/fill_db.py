from langchain_community.document_loaders import PyPDFDirectoryLoader
import chromadb
import os
import re
import json
from datetime import datetime
import shutil

DATA_PATH = "data"
CHROMA_DB_PATH = "chroma_db"
MEMORY_STORE_FILE = "memory/memory_store.json"

def detect_title(text):
    # Critères pour détecter un titre
    title_patterns = [
        r'^[A-Z\s]{2,}$',  # Texte tout en majuscules (modifié pour accepter 2 caractères minimum)
        r'^[0-9]+\.[0-9]*\s+[A-Z]',  # Titres numérotés (ex: 1.2 TITRE)
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Titres en CamelCase
        r'^[IVX]+\.\s+[A-Z]',  # Titres en chiffres romains
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:$',  # Titres suivis de deux points
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\([^)]*\)$',  # Titres avec parenthèses
    ]
    
    text = text.strip()
    for pattern in title_patterns:
        if re.match(pattern, text):
            return True
    return False

def process_conversations():
    """Traite les conversations du fichier memory_store.json et les prépare pour ChromaDB"""
    if not os.path.exists(MEMORY_STORE_FILE):
        return [], [], []

    with open(MEMORY_STORE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    # Traiter chaque conversation
    for i, conversation in enumerate(data):
        # Convertir la conversation en texte formaté
        document_text = f"Conversation ID: {conversation.get('id', '')}\n"
        document_text += f"Date: {conversation.get('timestamp', '')}\n"
        document_text += f"Utilisateur: {conversation.get('user_input', '')}\n"
        document_text += f"IA: {conversation.get('ai_response', '')}\n"
        document_text += f"Contexte: {conversation.get('context', '')}\n"
        document_text += f"Émotions détectées: {conversation.get('emotions', '')}\n"

        documents.append(document_text)
        metadatas.append({
            "source": MEMORY_STORE_FILE,
            "type": "conversation",
            "conversation_id": conversation.get('id', ''),
            "timestamp": conversation.get('timestamp', ''),
            "emotions": conversation.get('emotions', ''),
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        ids.append(f"conv_{i}")

    return documents, metadatas, ids

def update_database():
    """
    Met à jour la base de données Chroma avec les nouvelles conversations
    sans supprimer la base existante.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("documents")

    # Traiter uniquement les nouvelles conversations
    conv_documents, conv_metadatas, conv_ids = process_conversations()
    
    if conv_documents:
        # Récupérer les IDs existants
        existing_ids = set(collection.get()["ids"])
        
        # Filtrer pour ne garder que les nouvelles conversations
        new_documents = []
        new_metadatas = []
        new_ids = []
        
        for doc, meta, id in zip(conv_documents, conv_metadatas, conv_ids):
            if id not in existing_ids:
                new_documents.append(doc)
                new_metadatas.append(meta)
                new_ids.append(id)
        
        # Ajouter uniquement les nouvelles conversations
        if new_documents:
            collection.add(
                ids=new_ids,
                documents=new_documents,
                metadatas=new_metadatas,
            )

def add_documents_to_db():
    print("Initialisation de ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("documents")

    all_documents = []
    all_metadatas = []
    all_ids = []

    # Traiter les fichiers PDF
    print("Chargement des documents PDF...")
    loader = PyPDFDirectoryLoader(path=DATA_PATH)
    raw_documents = loader.load()

    for i, doc in enumerate(raw_documents):
        lines = doc.page_content.split('\n')
        titles = [line.strip() for line in lines if detect_title(line)]
        
        all_documents.append(doc.page_content)
        all_metadatas.append({
            "source": doc.metadata["source"],
            "type": "pdf",
            "page": doc.metadata.get("page", 0),
            "titles": ", ".join(titles) if titles else "Sans titre",
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        all_ids.append(f"pdf_{i}")

    # Traiter les conversations
    conv_documents, conv_metadatas, conv_ids = process_conversations()
    all_documents.extend(conv_documents)
    all_metadatas.extend(conv_metadatas)
    all_ids.extend(conv_ids)

    if all_documents:
        print("\nAjout des documents à la base de données...")
        collection.add(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
        )
        print(f"Ajouté {len(all_documents)} documents à la base de données")
    else:
        print("Aucun nouveau document trouvé")

if __name__ == "__main__":
    # Supprimer le dossier chroma_db s'il existe
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Suppression de l'ancienne base de données...")
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Base de données supprimée")
    
    # Créer le dossier data s'il n'existe pas
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Dossier {DATA_PATH} créé")
    
    print(f"Placez vos fichiers PDF dans le dossier '{DATA_PATH}' et assurez-vous que le fichier '{MEMORY_STORE_FILE}' est à jour, puis appuyez sur Entrée...")
    input()
    
    add_documents_to_db()
    print("\nTraitement terminé!")