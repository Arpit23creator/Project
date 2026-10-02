import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from chatbot import answer_help_query, summarize_meeting


app = FastAPI(
    title="VOW AI Engine",
    description="Python ML Microservice for VOW Help Desk & Meeting Intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HelpRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question asked by user")

class SummaryRequest(BaseModel):
    transcript: str = Field(..., min_length=5, description="Meeting transcript to summarize")

class AIResponse(BaseModel):
    reply: str
    status: str = "success"

@app.post("/api/ai/help", response_model=AIResponse)
async def handle_help_chat(request: HelpRequest):
    try:
        bot_response = answer_help_query(request.query)
        return AIResponse(reply=bot_response, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI inference error: {str(e)}")

@app.post("/api/ai/summarize", response_model=AIResponse)
async def handle_meeting_summary(request: SummaryRequest):
    try:
        summary_result = summarize_meeting(request.transcript)
        return AIResponse(reply=summary_result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)