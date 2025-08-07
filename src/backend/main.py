from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# --- REVISED IMPORTS ---
# We now use absolute imports from the 'src' directory, which is visible
# from the project root where uvicorn is running.
from src.analysis_engine.core import AnalysisEngine
# The AnalysisEngine's code imports PromptLinterModel, so Python needs to be
# able to find the `training` package as well.
from src.training.model import PromptLinterModel

# --- API Application Setup (Unchanged) ---
app = FastAPI(
    title="Prompt Linter",
    description="A pre-flight check for LLM prompts.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins, fine for local dev
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State Management (Unchanged) ---
state = {"engine": None}

# --- Pydantic Schemas (Unchanged) ---
class AnalysisRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=500, description="The user prompt to be analyzed.")

class AnalysisResponse(BaseModel):
    data: dict

# --- Application Events ---
@app.on_event("startup")
def startup_event():
    logger.info("--- Server is starting up ---")
    # This path is relative to the project root, so it remains correct.
    artifacts_path = "artifacts"
    logger.info(f"Initializing AnalysisEngine from: {artifacts_path}")
    state["engine"] = AnalysisEngine(artifacts_dir=artifacts_path)
    logger.info("--- AnalysisEngine initialization complete ---")

# --- API Endpoints (Unchanged) ---
@app.get("/", summary="Health check endpoint")
def read_root():
    return {"status": "ok", "message": "Guardrail API is running."}

@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze a user prompt")
def analyze_prompt(request: AnalysisRequest):
    if not state["engine"]:
        raise HTTPException(status_code=503, detail="Engine not available. Please try again later.")
    logger.info(f"Received analysis request for prompt: '{request.prompt}'")
    try:
        analysis_result = state["engine"].analyze(request.prompt)
        return {"data": analysis_result}
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during analysis.")