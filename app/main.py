from typing import Union

from pathlib import Path
import sys
import os

from fastapi import FastAPI, Request, Form
from db import queries as queries
from db import tfidf_search as tfidf
import numpy as np

# Ensure repository root is on sys.path so top-level packages (e.g. `scripts`)
# are importable when this module is loaded with the working directory set to
# the `app/` folder (fastapi dev may insert `app/` into sys.path).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.db.load_embeddings import load_embeddings
from app.db.embedding_search import embedding_search 
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import json
import types 


app = FastAPI()

# Templates directory is at project root, not in app/
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
    modelType: str= Form("")
): 
    keyword = message.lower().strip()
    print(modelType)
    # Convert credits to int (handle empty string)
    try:
        credits_int = int(credits) if credits else None
    except ValueError:
        credits_int = None
    
    # Convert bucket to int or None
    try:
        bucket_int = int(bucket) if bucket != "None" else None
    except ValueError:
        bucket_int = None
    
    # Build kwargs only with non-None/non-empty values
    search_kwargs = {"query": keyword}
    
    if bucket_int is not None:
        search_kwargs["bucketLevel"] = bucket_int
    
    if prefix:  # Only add if not empty string
        search_kwargs["subject"] = prefix
    
    if credits_int is not None:
        search_kwargs["credits"] = credits_int
    
    if major_requirement:  # Only add if not empty string
        search_kwargs["major_requirement"] = major_requirement.strip()
    
    # Search for courses with filters
    try:
        if modelType == "tfidf":
            result = tfidf.tool_search(**search_kwargs)
                # Extract the results list from the dictionary
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
            result=""
    except Exception as e:  
        return {"response": f'<div class="message-content"><p>Error: {str(e)}</p></div>'}
    
    print(result) 


    # Render template
    if classes:
        html_content = templates.get_template("ClassListTemplate.html").render({"classes": classes})
    else:
        html_content = '<div class="message-content"><p>No classes found matching your search. Try keywords like "algorithms", "data structures", or "programming".</p></div>'
    
    return {"response": html_content}

@app.get("/healthz")
def read_root():
    return {"status:ok", "db:"+str(len(queries.random_courses_safe())!=0)}

@app.get("/courses/sample")
def read_root():
    result=queries.random_courses_safe()
    return {json.dumps(result, default=my_object_encoder)}

@app.get("/courses/search/{name}")
def read_item(name: str, q: int= 10):
    result=queries.search_courses_by_title_safe(name, q)
    return {json.dumps(result, default=my_object_encoder)}

@app.get("/courses/search/{subject}/{id}")
def read_item(subject: str,id: int):
    result=queries.get_course_by_code_safe(subject,id)
    return {json.dumps(result, default=my_object_encoder)}


def my_object_encoder(obj):
    if isinstance(obj, types.SimpleNamespace):
        return {"subject": obj.subject, "number": obj.number, "title":obj.title, "snippet":obj.text}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    