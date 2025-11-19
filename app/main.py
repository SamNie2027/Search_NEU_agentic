from typing import Union

from pathlib import Path
import sys
import os

from fastapi import FastAPI
from db import queries as queries
import numpy as np

# Ensure repository root is on sys.path so top-level packages (e.g. `scripts`)
# are importable when this module is loaded with the working directory set to
# the `app/` folder (fastapi dev may insert `app/` into sys.path).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.db.load_embeddings import load_embeddings

import json
from types import SimpleNamespace

app = FastAPI()

# Loads once here 
courses, embeddings = load_embeddings()

@app.get("/")
def read_root():
    return {"Hello World"}


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
    return {"subject": obj.subject, "number": obj.number, "title":obj.title, "description":obj.description}