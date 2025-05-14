# tourism_agent_system/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tourism_agent_system.Agent.orchestrator import AgentOrchestrator
from tourism_agent_system.Agent.TrackingAgent import TrackingAgent
import os
import time
from typing import Dict, Any

app = FastAPI(title="Tourism Agent System API")

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # secondes

# 1) Activer CORS pour autoriser toutes les origines 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en dev : autorise tout ; en prod, liste tes domaines
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Instancier l'orchestrateur et le tracking agent une seule fois
orchestrator = AgentOrchestrator()
tracking_agent = TrackingAgent()

def handle_rate_limit(retry_count: int) -> None:
    """Gère le rate limiting en attendant avant de réessayer."""
    if retry_count < MAX_RETRIES:
        time.sleep(RETRY_DELAY * (retry_count + 1))  # Attente exponentielle
    else:
        raise Exception("Nombre maximum de tentatives atteint pour l'API Mistral")

@app.post("/chat")
def chat_endpoint(payload: dict) -> Dict[str, Any]:
    """
    payload attend : { "message": "Bonjour !" }
    Retourne : { "success": bool, "response": str, "error": str (optionnel) }
    """
    message = payload.get("message")
    if not message or not isinstance(message, str):
        return {
            "success": False,
            "response": "Il faut fournir un champ 'message' de type string.",
            "error": "Invalid input"
        }
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # Log de l'étape initiale
            tracking_agent.log_execution(
                agent_name="orchestrator",
                action="Réception de la demande utilisateur",
                status="succès"
            )
            
            # Appel de l'orchestrator
            result = orchestrator.process_message(message)
            
            # Log de l'interaction dans le tracking agent
            tracking_agent.log(
                agent_name="orchestrator",
                input_data=message,
                output_data=str(result)
            )
            
            # Log de l'étape finale
            tracking_agent.log_execution(
                agent_name="orchestrator",
                action="Génération de la réponse finale",
                status="succès"
            )
            
            # S'assurer que la réponse a la bonne structure
            if isinstance(result, dict):
                if "response" in result:
                    return {
                        "success": True,
                        "response": result["response"]
                    }
                else:
                    return {
                        "success": True,
                        "response": str(result)
                    }
            else:
                return {
                    "success": True,
                    "response": str(result)
                }
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:  # Rate limit
                retry_count += 1
                tracking_agent.log_execution(
                    agent_name="orchestrator",
                    action="Rate limit détecté, nouvelle tentative",
                    status=f"tentative {retry_count}/{MAX_RETRIES}"
                )
                handle_rate_limit(retry_count)
                continue
            else:
                # Log de l'erreur
                tracking_agent.log_execution(
                    agent_name="orchestrator",
                    action="Erreur lors du traitement",
                    status=f"erreur: {error_msg}"
                )
                return {
                    "success": False,
                    "response": "Désolé, une erreur est survenue lors du traitement de votre message.",
                    "error": error_msg
                }
    
    # Si on arrive ici, c'est que toutes les tentatives ont échoué
    return {
        "success": False,
        "response": "Désolé, le service est temporairement surchargé. Veuillez réessayer dans quelques instants.",
        "error": "Rate limit exceeded after multiple retries"
    }

@app.post("/clear-memory")
def clear_memory_endpoint() -> Dict[str, Any]:
    """
    Endpoint pour effacer la mémoire de l'orchestrateur
    """
    try:
        orchestrator.clear_memory()
        # Effacer aussi les logs du tracking agent
        tracking_agent.logs = []
        tracking_agent.execution_sequence = []
        return {"success": True, "message": "Mémoire et logs effacés avec succès"}
    except Exception as e:
        return {
            "success": False,
            "message": f"Erreur lors de l'effacement de la mémoire : {str(e)}",
            "error": str(e)
        }

@app.get("/analysis")
def get_analysis() -> Dict[str, Any]:
    """
    Endpoint pour obtenir l'analyse des interactions
    """
    try:
        analysis = tracking_agent.analyze_interactions()
        return analysis
    except Exception as e:
        return {
            "success": False,
            "message": f"Erreur lors de l'analyse : {str(e)}",
            "error": str(e)
        }

@app.get("/report")
def get_report() -> FileResponse:
    """
    Endpoint pour générer et télécharger le rapport d'analyse
    """
    try:
        # Générer le rapport
        report_path = "agent_analysis_report.md"
        tracking_agent.write_report(report_path)
        
        # Vérifier si le fichier existe
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Le rapport n'a pas pu être généré")
            
        # Retourner le fichier
        return FileResponse(
            path=report_path,
            filename="agent_analysis_report.md",
            media_type="text/markdown"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du rapport : {str(e)}")

@app.get("/logs")
def get_logs() -> Dict[str, Any]:
    """
    Endpoint pour obtenir les logs bruts
    """
    try:
        return {
            "success": True,
            "logs": tracking_agent.logs,
            "execution_sequence": tracking_agent.execution_sequence
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Erreur lors de la récupération des logs : {str(e)}",
            "error": str(e)
        }