from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(title="Stock Market Predictor API")

# In-memory dict to track background training status per ticker.
# Possible values: "training", "ready"
# Stored on app.state so all route handlers share the same dict.
app.state.training_status: dict[str, str] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
