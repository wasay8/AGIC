from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
app=Starlette()
path = Path(r'C:\Users\user\dataset-resized')
path.mkdir(parents=True, exist_ok=True)
path.ls()
classes = ['cardboard','glass','metal','paper','plastic','trash']
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", size=224, valid_pct=0.2,
        ds_tfms=get_transforms(),num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
#learn.fit_one_cycle(5)
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-3))
@app.route("/")
def form(request):
    return HTMLResponse("""
        <h3>This app will classify Cardboard vs Glass vs Trash vs Plastic vs Metal vs Paper<h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)
@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)
@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)
async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _, class_, losses = learn.predict(img)
    return JSONResponse({
        "prediction": classes[class_.item()],
        "scores": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })
if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=80)
