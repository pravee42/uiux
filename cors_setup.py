import subprocess

def install_cors():
    try:
        subprocess.check_call(["pip", "install", "starlette-cors"])
        print("starlette-cors installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing starlette-cors: {e}")

def configure_cors():
    cors_code = """
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://0.0.0.0:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}
"""
    with open("main.py", "w") as f:
        f.write(cors_code)
    print("CORS configured in main.py. Please replace your existing FastAPI app with this code.")

if __name__ == "__main__":
    install_cors()
    configure_cors()
