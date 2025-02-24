from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
#"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"


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
