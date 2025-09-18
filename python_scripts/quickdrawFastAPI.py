import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =============================== CNN 모델 정의
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1,128*3*3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# =============================== 클래스 & GI
classes = [
    "apple","bear","birthday cake","blackberry","blueberry","bread","broccoli",
    "cake","carrot","cookie","cow","crab","donut","duck","fish","grapes",
    "hamburger","hot dog","ice cream","lobster","lollipop","mushroom",
    "octopus","onion","peanut","pear","pig","pineapple","pizza","popsicle",
    "potato","sandwich","steak","strawberry","string bean","watermelon","wine"
]

GI_scores = {
    "apple":6,"bear":0,"birthday cake":23,"blackberry":2,"blueberry":3,"bread":20,
    "broccoli":1,"cake":22,"carrot":4,"cookie":19,"cow":0,"crab":0,"donut":21,"duck":0,
    "fish":0,"grapes":8,"hamburger":14,"hot dog":10,"ice cream":12,"lobster":0,"lollipop":20,
    "mushroom":1,"octopus":0,"onion":3,"peanut":1,"pear":5,"pig":0,"pineapple":13,
    "pizza":18,"popsicle":11,"potato":24,"sandwich":16,"steak":0,"strawberry":3,
    "string bean":1,"watermelon":9,"wine":2

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(len(classes)).to(device)
model.load_state_dict(torch.load("quickdraw_food_cnn_stable.pth", map_location=device))
model.eval()

# =============================== Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])

# =============================== FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/quickdraw/predict")
async def quickdraw_predict(image_base64: str = Form(...), current_glucose: float = Form(...)):
    try:
        # Base64 -> 이미지 처리 부분
        img_bytes = base64.b64decode(image_base64.split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes)).convert('L')

        # 반전 + 자동 크롭 + 중앙 정렬
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)  # 실제 그림만 크롭
        img = ImageOps.pad(img, (28,28), method=Image.Resampling.BILINEAR, color=0)  # 검은 배경
        arr = np.array(img).astype(np.float32)/255.0
        tensor_img = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)

        # Top-3 평균 20회
        probs_sum = torch.zeros(len(classes), device=device)
        with torch.no_grad():
            for _ in range(20):
                outputs = model(tensor_img)
                probs = F.softmax(outputs,dim=1)[0]
                top3 = torch.topk(probs,3).indices
                mask = torch.zeros_like(probs)
                mask[top3] = probs[top3]
                probs_sum += mask
        probs_avg = probs_sum / 20
        recognized_class = classes[probs_avg.argmax().item()]
        sugar_increase = GI_scores.get(recognized_class,0)
        final_glucose = current_glucose + sugar_increase

        return JSONResponse({
            "recognized_class": recognized_class,
            "sugar_increase": sugar_increase,
            "final_glucose": final_glucose
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101, log_level="warning")
