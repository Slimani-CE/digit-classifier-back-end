from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from pydantic import BaseModel
import cv2
import uvicorn
import numpy as np
import pickle


# Classifier Model Class. This will be the definition of the classifier that
# Will be importing

class Result(BaseModel):
    digit: int
    proba: float


class DigitClassifier:
    def __init__(self, model, IMG_SIZE):
        self.model = model
        self.IMG_SIZE = IMG_SIZE

    def predict(self, img):
        img = img[:, :, -1]

        resized = cv2.resize(
            img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)
        newImg = resized / 255
        newImg = np.array(newImg).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

        prediction = self.model.predict(newImg)

        return str(np.argmax(prediction))


# Import the classifier using pickle
with open('model/digit-classifier-model.pkl', 'rb') as file:
    classifier = pickle.load(file)

app = FastAPI()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post('/digit-classifier-api')
async def upload_image(image: UploadFile = File(...)):
    print("NEW REQUEST")
    contents = await image.read()
    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    prediction = classifier.predict(img)
    print("New image classified as: ", prediction)

    result = Result(digit=prediction, proba=-1)

    return result

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8082,
                ssl_keyfile='./key.pem', ssl_certfile='./cert.pem')
