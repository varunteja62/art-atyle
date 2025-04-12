from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import torch
import io
import os
import uuid

 
app = FastAPI()

 


# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

@app.get("/get")
async def sample_api():
    return "Backend is working 🎉"

def stylize_image(content_image: Image.Image, style: str):
    model_paths = {
        "madhubani": "models/madhubani (3).pth",
        "kalamkari": "models/kalamkari.pth",
        "warli": "models/warli.pth",
    }

    if style not in model_paths:
        raise ValueError("Invalid style selected.")

    # Load the model
    model = torch.jit.load(model_paths[style]).eval()

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_tensor = transform(content_image).unsqueeze(0)

    with torch.no_grad():
        output = model(content_tensor).cpu()

    output_image = transforms.ToPILImage()(output.squeeze() / 255)
    return output_image

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), style: str = "madhubani"):
    try:
        print(f"📥 Received file: {file.filename}, Style: {style}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        output_img = stylize_image(image, style)

        output_filename = f"output_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join("output", output_filename)

        output_img.save(output_path)

        print(f"✅ Saved stylized image to: {output_path}")
        return FileResponse(output_path, media_type="image/jpeg")

    except Exception as e:
        print(f"🔥 Error occurred: {str(e)}")  # <=== IMPORTANT!
        return JSONResponse(status_code=500, content={"error": str(e)})
