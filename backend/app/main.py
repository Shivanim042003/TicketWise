from fastapi import FastAPI

app = FastAPI(
    title="TicketWise API",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "Welcome to TicketWise API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
