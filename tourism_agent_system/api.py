# tourism_agent_system/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tourism_agent_system.Agent.orchestrator import AgentOrchestrator


app = FastAPI(title="Tourism Agent System API")
/
# 1) Activer CORS pour autoriser toutes les origines 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en dev : autorise tout ; en prod, liste tes domaines
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Instancier l'orchestrateur une seule fois
orchestrator = AgentOrchestrator()

@app.post("/chat")
def chat_endpoint(payload: dict):
    """
    payload attend : { "message": "Bonjour !" }
    """
    message = payload.get("message")
    if not message or not isinstance(message, str):
        raise HTTPException(status_code=400, detail="Il faut fournir un champ 'message' de type string.")
    # Appel de ton orchestrator
    result = orchestrator.process_message(message)
    return result