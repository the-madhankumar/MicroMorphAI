from fastapi import FastAPI

app = FastAPI()

RANDOM_FOREST = ""
EMBEDDINGS_COLLECTION = ""
YOLO = ""
MASK_R_CNN = ""
SAM = ""


@app.get("/dhatchayani")
async def root():
    return {"message": "Hello World"}