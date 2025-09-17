import torch
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch import nn
from PIL import Image
import pandas as pd
import numpy as np
import os
import shutil
import re
import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import matplotlib.pyplot as plt

# ===============================
# 프로젝트 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# 1 클래스 불러오기
flatten_dir = os.path.join(BASE_DIR, "kfood_flat")
dataset = datasets.ImageFolder(root=flatten_dir)
classes = dataset.classes

# ===============================
# 2 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model = model.to(device)

model_path = os.path.join(BASE_DIR, "food_classifier_resnet18_sampled.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===============================
# 3 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 4 음식 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# ===============================
# 5 CSV 불러오기
csv_path = os.path.join(BASE_DIR, "food_with_GI_GL_2.csv")
df = pd.read_csv(csv_path, encoding="cp949")

def get_nutrition(food_name):
    row = df[df["음식명"] == food_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# ===============================
# 6 안전한 파일명
def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9]', '', name)

# ===============================
# 7 혈당 곡선 + Base64 반환
def plot_glucose_curve_base64(food_name, score, GL, baseline=90):
    time = np.linspace(0, 120, 50)
    peak_time = 40
    sigma = 20
    max_rise = score * (GL / 10)
    curve = baseline + max_rise * np.exp(-((time - peak_time) ** 2) / (2 * sigma ** 2))
    max_glucose = curve.max()

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(7,4))
    plt.plot(time, curve, color="red", linewidth=2, label="예상 혈당(mg/dL)")
    plt.fill_between(time, baseline, curve, color="red", alpha=0.2)
    plt.axhline(y=baseline, color="blue", linestyle="--", label=f"현재 혈당 {baseline} mg/dL")
    plt.title(f"{food_name} 예상 혈당 반응 (2시간)")
    plt.xlabel("시간 (분)")
    plt.ylabel("혈당 (mg/dL)")
    plt.legend()
    plt.grid(True)

    # 임시 저장
    safe_name = sanitize_filename(food_name)
    temp_path = os.path.join(os.getenv("TEMP", "/tmp"), f"glucose_{safe_name}.png")
    plt.savefig(temp_path, format="png", dpi=150)
    plt.close()

    # Base64 인코딩
    with open(temp_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return max_glucose, img_base64

# ===============================
# 8 운동 추천
def recommend_exercise(baseline, max_glucose):
    rise = max_glucose - baseline
    if rise >= 50:
        return ("혈당이 많이 상승했습니다.\n"
                "- 식사 후 20~30분 내 빠르게 걷기 15~20분\n"
                "- 가벼운 자전거 타기 10~15분\n"
                "- 가능하면 계단 오르기 5~10분")
    elif rise >= 30:
        return ("혈당이 조금 상승했습니다.\n"
                "- 식사 후 30~50분 정도 걷기 10~15분\n"
                "- 간단한 스트레칭 5~10분")
    else:
        return ("혈당 변화가 적습니다.\n"
                "- 식사 후 50~60분 가벼운 근력 운동 10~15분\n"
                "- 일상 활동(청소, 집안일) 등으로 소모")

# ===============================
# 9 FastAPI 서버
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/python/predict")
async def predict(file: UploadFile, currentGlucose: str = Form("90")):
    try:
        try:
            baseline = float(currentGlucose)
        except:
            baseline = 90.0

        temp_path = os.path.join(os.getenv("TEMP", "/tmp"), file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        food = predict_image(temp_path)
        nutrition = get_nutrition(food)

        if nutrition:
            GI = nutrition["추정_GI"]
            GL = nutrition["추정_GL"]
            score = nutrition["혈당영향점수"]

            max_glucose, img_base64 = plot_glucose_curve_base64(food, score, GL, baseline)
            exercise_text = recommend_exercise(baseline, max_glucose)

            text_result = f"예측 음식: {food}\n영양 정보:\n"
            for k, v in nutrition.items():
                text_result += f" - {k}: {v}\n"
            text_result += f"예상 최대 혈당: {max_glucose:.1f} mg/dL\n"
            text_result += f"운동 추천: {exercise_text}"

            return JSONResponse({
                "textResult": text_result,
                "curveImageBase64": img_base64
            })
        else:
            return JSONResponse({
                "textResult": f"CSV에서 {food} 영양 정보를 찾을 수 없습니다.",
                "curveImageBase64": None
            })
    except Exception as e:
        return JSONResponse({
            "textResult": f"서버 에러 발생: {str(e)}",
            "curveImageBase64": None
        })

# ===============================
# 10 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
