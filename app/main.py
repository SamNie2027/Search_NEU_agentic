"""
FastAPI application for course search.

Provides web interface and API endpoints for searching courses using
TF-IDF, embeddings, and ReAct agent-based search.
"""
from pathlib import Path
import sys
import json
import types

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.database import queries
from app.search import tfidf_search as tfidf
from app.agent import run_agent
from app.search.load_embeddings import load_embeddings
from app.search.embedding_search import embedding_search

app = FastAPI()
templates = Jinja2Templates(directory=str(repo_root / "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("ChatWindowTemplate.html", {
        "request": request
    })

@app.post("/api/chat", response_class=JSONResponse)
async def chat(
    message: str = Form(...),
    prefix: str = Form(""),
    bucket: str = Form("None"),
    credits: str = Form(""),
    major_requirement: str = Form(""),
    modelType: str = Form(""),
    isChecked: str = Form("false"),
    useFilters: str = Form("0"),
):
    """
    Handle course search requests from the frontend.
    
    Supports TF-IDF, embeddings, and agent-based search with optional filters.
    Accepts both legacy `isChecked` and newer `useFilters` parameters for backward compatibility.
    """
    keyword = message.lower().strip()
    
    def _to_bool(val: str) -> bool:
        """Convert string to boolean, accepting various truthy values."""
        if val is None:
            return False
        v = str(val).strip().lower()
        return v in ("1", "true", "t", "yes", "y", "on")

    use_filters = _to_bool(useFilters) or _to_bool(isChecked)
    
    try:
        credits_int = int(credits) if credits else None
    except ValueError:
        credits_int = None
    
    try:
        bucket_int = int(bucket) if bucket != "None" else None
    except ValueError:
        bucket_int = None
    
    search_kwargs = {"query": keyword}
    
    if bucket_int is not None:
        search_kwargs["bucketLevel"] = bucket_int
    
    if prefix:
        search_kwargs["subject"] = prefix
    
    if credits_int is not None:
        search_kwargs["credits"] = credits_int
    
    if major_requirement:
        search_kwargs["major_requirement"] = major_requirement.strip()
    
    try:
        if modelType == "tfidf":
            result = tfidf.tool_search(**search_kwargs)
            if result and isinstance(result, dict) and 'results' in result:
                classes = result['results']
            elif result:
                classes = result
            else:
                classes = []
        elif modelType == "embeddings":
            courses, embeddings = load_embeddings()
            result = embedding_search(keyword, courses, embeddings)
    
            if result:
                classes = []
                for item in result:
                    new_item = item.copy() 
                    new_item['snippet'] = new_item.pop('text') 
                    classes.append(new_item)
            else:
                classes = []
        elif modelType == "agent":
            print("useFilters", use_filters)
            result=run_agent.run_agent_with_real_llm(keyword,6, useFilters=use_filters)
            print(result)
            if result and isinstance(result, dict) and 'results' in result:
                classes = result['results']
            elif result:
                classes = result
            else:
                classes = []
                 
    except Exception as e:  
        return {"response": f'<div class="message-content"><p>Error: {str(e)}</p></div>'}
    
    print(result) 

    if classes:
        html_content = templates.get_template("ClassListTemplate.html").render({"classes": classes})
    else:
        html_content = '<div class="message-content"><p>No classes found matching your search. Try keywords like "algorithms", "data structures", or "programming".</p></div>'
    
    return {"response": html_content}

@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    db_available = len(queries.random_courses_safe()) != 0
    return {"status": "ok", "db": str(db_available)}


@app.get("/courses/sample")
def sample_courses():
    """Get a sample of random courses."""
    result = queries.random_courses_safe()
    return JSONResponse(content=json.loads(json.dumps(result, default=my_object_encoder)))


@app.get("/courses/search/{name}")
def search_courses_by_name(name: str, q: int = 10):
    """Search courses by title."""
    result = queries.search_courses_by_title_safe(name, q)
    serialized = json.dumps(result, default=my_object_encoder)
    return JSONResponse(content=json.loads(serialized))


@app.get("/courses/search/{subject}/{id}")
def get_course(subject: str, id: int):
    """Get a specific course by subject and number."""
    result = queries.get_course_by_code_safe(subject, id)
    serialized = json.dumps(result, default=my_object_encoder)
    return JSONResponse(content=json.loads(serialized))


def my_object_encoder(obj):
    """JSON encoder for SimpleNamespace objects."""
    if isinstance(obj, types.SimpleNamespace):
        return {"subject": obj.subject, "number": obj.number, "title": obj.title, "snippet": obj.text}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    