import streamlit as st
from PIL import Image
import requests
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Tourism Agent System", page_icon="🌍", layout="wide")

# Titre et sous-titre
st.title("Tourism Agent System")
st.subheader("Bienvenue dans le système d'agent touristique!")

# Initialisation de l'historique des messages dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fonction pour envoyer un message
def send_message(message):
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": message},
            timeout=10  # Ajout d'un timeout
        )
        
        if response.status_code != 200:
            st.error(f"Erreur serveur (code {response.status_code})")
            return {
                "response": "Erreur de communication avec le serveur",
                "success": False
            }
            
        response_data = response.json()
        
        if not response_data.get("success", False):
            error_msg = response_data.get("error", "Une erreur est survenue")
            st.error(error_msg)
            return {
                "response": error_msg,
                "success": False
            }
            
        return {
            "response": response_data.get("response", "Pas de réponse du serveur"),
            "success": True
        }
    except requests.exceptions.Timeout:
        st.error("Le serveur met trop de temps à répondre")
        return {
            "response": "Délai d'attente dépassé",
            "success": False
        }
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter au serveur")
        return {
            "response": "Erreur de connexion au serveur",
            "success": False
        }
    except Exception as e:
        st.error(f"Erreur inattendue : {str(e)}")
        return {
            "response": "Une erreur inattendue est survenue",
            "success": False
        }

# Fonction pour effacer la mémoire
def clear_memory():
    response = requests.post("http://localhost:8000/clear-memory")
    if response.json()["success"]:
        st.session_state.messages = []
        st.success("Mémoire effacée avec succès!")

# Fonction pour obtenir l'analyse
def get_analysis():
    try:
        response = requests.get("http://localhost:8000/analysis")
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'analyse : {str(e)}")
        return None

# Fonction pour obtenir les logs
def get_logs():
    try:
        response = requests.get("http://localhost:8000/logs")
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des logs : {str(e)}")
        return None

# Fonction pour télécharger le rapport
def download_report():
    try:
        response = requests.get("http://localhost:8000/report")
        if response.status_code == 200:
            # Sauvegarder le rapport
            filename = f"agent_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        else:
            st.error("Erreur lors de la génération du rapport")
            return None
    except Exception as e:
        st.error(f"Erreur lors du téléchargement du rapport : {str(e)}")
        return None

# Création de deux colonnes pour l'interface
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de chat principale
    st.markdown("### 💬 Chat")
    
    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Zone de saisie du message
    if prompt := st.chat_input("Écrivez votre message ici..."):
        try:
            # Ajout du message de l'utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Envoi du message et récupération de la réponse
            response = send_message(prompt)
            
            # Vérification de la réponse
            if isinstance(response, dict):
                response_text = response.get("response", "Erreur de communication avec le serveur")
                if not response.get("success", False):
                    st.error("Erreur lors de la communication avec le serveur")
            else:
                response_text = "Erreur de communication avec le serveur"
                st.error("Format de réponse invalide")
            
            # Affichage de la réponse
            with st.chat_message("assistant"):
                st.write(response_text)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
        except Exception as e:
            error_msg = f"Erreur lors du traitement du message : {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Désolé, une erreur est survenue lors du traitement de votre message."
            })

    # Bouton pour effacer la mémoire
    if st.button("Effacer l'historique", type="secondary"):
        clear_memory()

with col2:
    # Section d'analyse et de monitoring
    st.markdown("### 📊 Analyse et Monitoring")
    
    # Bouton pour rafraîchir l'analyse
    if st.button("🔄 Rafraîchir l'analyse"):
        analysis = get_analysis()
        if analysis and analysis.get("status") == "success":
            st.markdown("#### Analyse des Interactions")
            st.markdown(analysis.get("analysis", "Pas d'analyse disponible"))
        else:
            st.warning("Aucune analyse disponible pour le moment")
    
    # Bouton pour télécharger le rapport
    if st.button("📥 Télécharger le rapport"):
        filename = download_report()
        if filename:
            st.success(f"Rapport téléchargé : {filename}")
    
    # Section des logs
    st.markdown("#### 📝 Logs récents")
    logs = get_logs()
    if logs and logs.get("success"):
        for log in logs.get("logs", [])[-5:]:  # Afficher les 5 derniers logs
            with st.expander(f"{log['agent']} - {log['timestamp']}"):
                st.markdown(f"**Input:** {log['input']}")
                st.markdown(f"**Output:** {log['output']}")
    else:
        st.info("Aucun log disponible")

# Ajout d'un footer
st.markdown("---")
st.markdown("### 📈 Statistiques")
if logs and logs.get("success"):
    total_logs = len(logs.get("logs", []))
    st.metric("Nombre total d'interactions", total_logs)
else:
    st.metric("Nombre total d'interactions", 0)

