from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


#modify when we have db access to query connection
@app.get("/healthz")
def read_root():
    return {"status:ok", "db:"+str(True)}

#modify when we can call db
@app.get("/courses/sample")
def read_root():
    return {"result of random_courses"}

#modify when we can call db
@app.get("/courses/search/{name}")
def read_item(name: str, q: int= 10):
    return {"name": name, "limit": q}