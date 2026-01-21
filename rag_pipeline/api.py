"""FastAPI service for smart contract vulnerability detection."""

import os
import time
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, init_db, RequestHistory
from rag_agent import VulnerabilityRAGAgent

# Initialize FastAPI app
app = FastAPI(
    title="Smart Contract Vulnerability Detector",
    description="RAG-based vulnerability detection for Solidity contracts",
    version="1.0.0"
)

# Auth token for delete operations
DELETE_TOKEN = os.getenv("DELETE_TOKEN", "supersecrettoken123")

# Lazy-load the agent (expensive to initialize)
_agent = None

def get_agent() -> VulnerabilityRAGAgent:
    global _agent
    if _agent is None:
        _agent = VulnerabilityRAGAgent()
    return _agent


#  Request/Response Models

class ForwardRequest(BaseModel):
    """Request body for /forward endpoint."""
    contract_code: str
    top_k: int = 5  # number of similar vulnerabilities to retrieve


class ForwardResponse(BaseModel):
    """Response body for /forward endpoint."""
    analysis: str
    processing_time_ms: float
    contract_length: int
    token_count: int


class HistoryItem(BaseModel):
    """Single history entry."""
    id: int
    timestamp: str
    contract_length: int
    token_count: int
    processing_time_ms: float
    status: str
    result: str | None
    error_message: str | None

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    """Statistics response."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    processing_time: dict # mean, p50, p95, p99
    input_stats: dict # avg length, avg tokens


# Helper Functions 

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English/code)."""
    return len(text) // 4


# Startup 

@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


# Endpoints 

@app.post("/forward", response_model=ForwardResponse)
def forward(request: ForwardRequest, db: Session = Depends(get_db)):
    """
    Analyze a smart contract for vulnerabilities.
    
    Accepts contract code in JSON format, returns vulnerability analysis.
    """
    # Validate input
    if not request.contract_code or not request.contract_code.strip():
        raise HTTPException(status_code=400, detail="bad request")
    
    contract_code = request.contract_code.strip()
    contract_length = len(contract_code)
    token_count = estimate_tokens(contract_code)
    
    # Process with timing
    start_time = time.time()
    
    try:
        agent = get_agent()
        analysis = agent.analyze_contract(contract_code, top_k=request.top_k)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Save to history
        history_entry = RequestHistory(
            contract_code=contract_code[:1000],  # truncate for storage
            contract_length=contract_length,
            token_count=token_count,
            processing_time_ms=processing_time_ms,
            status="success",
            result=analysis[:2000] if analysis else None  # truncate
        )
        db.add(history_entry)
        db.commit()
        
        return ForwardResponse(
            analysis=analysis,
            processing_time_ms=round(processing_time_ms, 2),
            contract_length=contract_length,
            token_count=token_count
        )
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Save error to history
        history_entry = RequestHistory(
            contract_code=contract_code[:1000],
            contract_length=contract_length,
            token_count=token_count,
            processing_time_ms=processing_time_ms,
            status="error",
            error_message=str(e)[:500]
        )
        db.add(history_entry)
        db.commit()
        
        raise HTTPException(
            status_code=403, 
            detail="модель не смогла обработать данные"
        )


@app.get("/history", response_model=list[HistoryItem])
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    """Get history of all requests."""
    history = db.query(RequestHistory)\
        .order_by(RequestHistory.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return [
        HistoryItem(
            id=h.id,
            timestamp=h.timestamp.isoformat(),
            contract_length=h.contract_length,
            token_count=h.token_count,
            processing_time_ms=h.processing_time_ms,
            status=h.status,
            result=h.result,
            error_message=h.error_message
        )
        for h in history
    ]


@app.delete("/history")
def delete_history(
    x_auth_token: str = Header(..., alias="X-Auth-Token"),
    db: Session = Depends(get_db)
):
    """
    Delete all request history.
    
    Requires X-Auth-Token header for authorization.
    """
    if x_auth_token != DELETE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    deleted_count = db.query(RequestHistory).delete()
    db.commit()
    
    return {"message": f"Deleted {deleted_count} records"}


@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get statistics about requests."""
    # Total counts
    total = db.query(func.count(RequestHistory.id)).scalar() or 0
    successful = db.query(func.count(RequestHistory.id))\
        .filter(RequestHistory.status == "success").scalar() or 0
    failed = total - successful
    
    if total == 0:
        return StatsResponse(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            processing_time={"mean": 0, "p50": 0, "p95": 0, "p99": 0},
            input_stats={"avg_length": 0, "avg_tokens": 0}
        )
    
    # Processing time stats
    times = [r.processing_time_ms for r in db.query(RequestHistory.processing_time_ms).all()]
    times.sort()
    
    def percentile(data: list, p: float) -> float:
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]
    
    processing_stats = {
        "mean": round(sum(times) / len(times), 2),
        "p50": round(percentile(times, 50), 2),
        "p95": round(percentile(times, 95), 2),
        "p99": round(percentile(times, 99), 2)
    }
    
    # Input stats
    avg_length = db.query(func.avg(RequestHistory.contract_length)).scalar() or 0
    avg_tokens = db.query(func.avg(RequestHistory.token_count)).scalar() or 0
    
    input_stats = {
        "avg_length": round(float(avg_length), 2),
        "avg_tokens": round(float(avg_tokens), 2)
    }
    
    return StatsResponse(
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        processing_time=processing_stats,
        input_stats=input_stats
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

