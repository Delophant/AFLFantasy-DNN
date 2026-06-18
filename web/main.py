from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

_DIR = Path(__file__).resolve().parent
_INDEX_HTML = (_DIR / "index.html").read_text(encoding="utf-8")
_GAME_HTML = (_DIR / "game.html").read_text(encoding="utf-8")

app = FastAPI(title="AFL Predictions")


@app.get("/", response_class=HTMLResponse)
async def index():
    return _INDEX_HTML


@app.get("/game", response_class=HTMLResponse)
async def game():
    return _GAME_HTML


@app.get("/data/predictions/{filename}")
async def serve_prediction(filename: str):
    path = _DIR / "data" / "predictions" / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
