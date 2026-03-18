from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import symptom, drug

app = FastAPI(title="AI Smart Health Guardian API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(symptom.router)
app.include_router(drug.router)

@app.get("/")
def root():
    return {"status": "Health Guardian API is running"}

