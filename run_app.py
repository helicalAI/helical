import uvicorn

HOST="0.0.0.0"
PORT=8000

if __name__ == "__main__":
    uvicorn.run("app.app:API", reload=True, host=HOST, port=PORT)