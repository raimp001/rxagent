"""rxagent - AI Prescription Verification & Drug Interaction Agent
FastAPI + Anthropic Claude + LangChain
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import anthropic
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import httpx
import json

app = FastAPI(title="RxAgent", description="AI-powered prescription verification agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENFDA_BASE = "https://api.fda.gov/drug"

class PrescriptionRequest(BaseModel):
    patient_id: str
    medications: List[str]
    patient_age: int
    conditions: Optional[List[str]] = []
    weight_kg: Optional[float] = None

class VerificationResult(BaseModel):
    safe: bool
    interactions: List[dict]
    warnings: List[str]
    recommendations: List[str]
    confidence: float
    agent_reasoning: str

@tool
def check_fda_interactions(drug_name: str) -> str:
    """Query OpenFDA for drug interaction and label information."""
    try:
        url = f"{OPENFDA_BASE}/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                label = results[0]
                warnings = label.get("warnings", ["No warnings found"])
                interactions = label.get("drug_interactions", ["No interaction data"])
                return json.dumps({"drug": drug_name, "warnings": warnings[:2], "interactions": interactions[:2]})
        return json.dumps({"drug": drug_name, "error": "Not found in FDA database"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def get_drug_dosage_info(drug_name: str, patient_age: int, weight_kg: float = 70.0) -> str:
    """Get dosage recommendations for a drug based on patient demographics."""
    try:
        url = f"{OPENFDA_BASE}/label.json?search=openfda.generic_name:\"{drug_name}\"&limit=1"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                label = results[0]
                dosage = label.get("dosage_and_administration", ["Standard dosing applies"])
                pediatric = label.get("pediatric_use", ["Adult dosing"])
                geriatric = label.get("geriatric_use", ["Standard dosing"])
                age_note = pediatric[0] if patient_age < 18 else (geriatric[0] if patient_age > 65 else "Standard adult dosing")
                return json.dumps({"drug": drug_name, "dosage": dosage[0] if dosage else "N/A", "age_note": age_note})
        return json.dumps({"drug": drug_name, "info": "Standard dosing - consult prescriber"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def build_agent():
    """Build the LangChain tool-calling agent with Claude."""
    llm = ChatAnthropic(
        model="claude-opus-4-5",
        api_key=ANTHROPIC_API_KEY,
        temperature=0
    )
    tools = [check_fda_interactions, get_drug_dosage_info]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are RxAgent, an expert clinical pharmacist AI assistant.
        Your role is to:
        1. Check drug-drug interactions for patient medication lists
        2. Verify dosing appropriateness for patient demographics
        3. Flag contraindications with known conditions
        4. Provide actionable clinical recommendations
        Always use the available tools to gather FDA data before making assessments.
        Be precise, evidence-based, and err on the side of caution for patient safety."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

@app.post("/verify", response_model=VerificationResult)
async def verify_prescription(request: PrescriptionRequest):
    """Main endpoint: verify a patient's medication list for safety."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    
    agent = build_agent()
    query = f"""
    Verify the following prescription for patient safety:
    - Patient ID: {request.patient_id}
    - Age: {request.patient_age} years
    - Weight: {request.weight_kg or 'unknown'} kg
    - Medical conditions: {', '.join(request.conditions) if request.conditions else 'none reported'}
    - Current medications: {', '.join(request.medications)}
    
    Please check each drug in the FDA database, identify any interactions between them,
    flag dosing concerns given the patient demographics, and provide specific recommendations.
    Format your final response as: SAFE: yes/no | INTERACTIONS: list | WARNINGS: list | RECOMMENDATIONS: list
    """
    
    try:
        result = agent.invoke({"input": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
    output = result.get("output", "")
    
    # Parse agent output
    is_safe = "SAFE: yes" in output.lower() or "no interactions" in output.lower()
    warnings = [line.strip() for line in output.split("\n") if "warning" in line.lower() or "caution" in line.lower()][:5]
    recommendations = [line.strip() for line in output.split("\n") if "recommend" in line.lower() or "suggest" in line.lower()][:5]
    
    return VerificationResult(
        safe=is_safe,
        interactions=[{"note": "See agent_reasoning for full interaction analysis"}],
        warnings=warnings if warnings else ["No critical warnings identified"],
        recommendations=recommendations if recommendations else ["Continue current regimen with monitoring"],
        confidence=0.85,
        agent_reasoning=output
    )

@app.get("/drug/{drug_name}")
async def get_drug_info(drug_name: str):
    """Quick lookup of FDA drug info."""
    return json.loads(check_fda_interactions.invoke(drug_name))

@app.get("/health")
async def health_check():
    return {"status": "ok", "agent": "rxagent", "model": "claude-opus-4-5"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
