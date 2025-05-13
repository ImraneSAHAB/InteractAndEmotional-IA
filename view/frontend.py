import streamlit as st
from PIL import Image
import requests

# Configuration de la page
st.set_page_config(page_title="Tourism Agent System", page_icon="🌍")

st.title("Tourism Agent System")
st.subheader("Bienvenue dans le système d'agent touristique!")

# Initialisation de l'historique des messages dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fonction pour envoyer un message
def send_message(message):
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": message}
    )
    return response.json()

# Fonction pour effacer la mémoire
def clear_memory():
    response = requests.post("http://localhost:8000/clear-memory")
    if response.json()["success"]:
        st.session_state.messages = []
        st.success("Mémoire effacée avec succès!")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie du message
if prompt := st.chat_input("Écrivez votre message ici..."):
    # Ajout du message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Envoi du message et récupération de la réponse
    response = send_message(prompt)
    
    # Affichage de la réponse
    with st.chat_message("assistant"):
        st.write(response["response"])
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})

# Bouton pour effacer la mémoire
if st.button("Effacer l'historique", type="secondary"):
    clear_memory()

