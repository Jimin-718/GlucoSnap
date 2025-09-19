const chat = document.getElementById('chat');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const photoBtn = document.getElementById('photoBtn');
const imageInput = document.getElementById('imageInput');

// 그림판 관련 DOM
const drawContainer = document.getElementById('draw-container');
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const submitDrawing = document.getElementById('submitDrawing');

let mode = "menu";
let drawing = false;

// 메시지 추가 함수
function addMessage(msg, sender) {
    const div = document.createElement('div');
    div.className = "message " + sender;
    div.innerHTML = msg;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// 초기 로드
window.onload = () => {
    drawContainer.style.display = "none";
    addMessage(
        "안녕하세요 무엇을 도와드릴까요?<br>" +
        "1. 음식 사진을 업로드하면 혈당 변화를 알려드려요.<br>" +
        "2. 그림판에 그림을 그리면 그림을 인식해 혈당 변화를 알려드려요.",
        "bot"
    );
};

// 이벤트 등록
sendBtn.addEventListener("click", handleSend);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSend();
});

function handleSend() {
    const msg = userInput.value.trim();
    if (!msg) return;
    addMessage(msg, "user");
    userInput.value = "";
    handleUserInput(msg);
}

// ================= 메뉴 선택 처리 =================
function handleUserInput(msg) {
    if (mode === "menu") {
        if (msg === "1") {
            fetch("/fastapi/start/1", { method: "POST" });
            addMessage("사진인식 서버 실행됨. 현재 본인의 혈당값을 입력 한 후, 사진을 업로드해주세요.", "bot");
            userInput.placeholder = "혈당값을 입력하세요...";
            photoBtn.style.display = "inline-block";
            mode = "blood";
        } else if (msg === "2") {
            fetch("/fastapi/start/2", { method: "POST" });
            addMessage("그림판 서버 실행됨. 미니게임 시작! 음식을 그려보세요.", "bot");
            drawContainer.style.display = "block";
            mode = "game";
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            if (!canvas.dataset.initialized) {
                canvas.addEventListener("mousedown", () => drawing = true);
                canvas.addEventListener("mouseup", () => drawing = false);
                canvas.addEventListener("mouseout", () => drawing = false);
                canvas.addEventListener("mousemove", draw);
                canvas.dataset.initialized = true;
            }
        } else {
            addMessage("메뉴는 1~2 중에서 선택해주세요.", "bot");
        }
    } else if (mode === "blood") {
        if (!isNaN(msg)) {
            addMessage(`현재 혈당 ${msg} mg/dL 기록 완료.`, "bot");
        } else {
            addMessage("혈당값은 숫자로 입력해주세요.", "bot");
        }
    }
}

// ================= 사진 업로드 처리 =================
photoBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
    if (!imageInput.files[0]) return;

    const file = imageInput.files[0];
    const imgURL = URL.createObjectURL(file);
    addMessage(`<b>업로드한 사진:</b><br><img src="${imgURL}" width="200">`, "user");
    addMessage("이미지 업로드 중...", "user");

    const formData = new FormData();
    formData.append("file", file);

    if (userInput.value && !isNaN(userInput.value)) {
        formData.append("currentGlucose", userInput.value);
        addMessage(`현재 혈당 ${userInput.value} mg/dL 기록 완료.`, "bot");
        userInput.value = "";
    }

    try {
        const res = await fetch("http://localhost:8100/python/predict", { method: "POST", body: formData });
        const text = await res.text();
        let data;
        try { data = JSON.parse(text); } catch (e) { data = { textResult: text }; }

        if (data.textResult)
            addMessage("<b>예측 결과:</b><br>" + data.textResult.replace(/\n/g, "<br>"), "bot");

        if (data.curveImageBase64)
            addMessage(`<img src="data:image/png;base64,${data.curveImageBase64}" width="400">`, "bot");
    } catch (err) {
        addMessage("오류 발생: " + err, "bot");
    }
});

// ================= 그림판 로직 =================
let lastX = 0, lastY = 0;

function draw(e) {
    if (!drawing) return;
    ctx.strokeStyle = "black";
    ctx.lineWidth = 8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    lastX = e.offsetX;
    lastY = e.offsetY;
}

canvas.addEventListener("mousedown", (e) => { drawing = true; lastX = e.offsetX; lastY = e.offsetY; });
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseout", () => drawing = false);
canvas.addEventListener("mousemove", draw);

clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// ================= 그림판 제출 & FastAPI 연동 =================
submitDrawing.addEventListener("click", async () => {
    canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onload = async () => {
            const formData = new FormData();
            formData.append("image_base64", reader.result);
            formData.append("current_glucose", userInput.value || 100);

            try {
                const res = await fetch("http://localhost:8101/quickdraw/predict", { method: "POST", body: formData });
                const text = await res.text();
                let data;
                try { data = JSON.parse(text); } catch (e) { data = { textResult: text }; }

                if (data.recognized_class || data.textResult) {
                    const msg = data.recognized_class
                        ? `✏그림 인식 결과: ${data.recognized_class}<br>혈당 증가량: ${data.sugar_increase || "N/A"} mg/dL<br>`
                        : `<b>서버 응답:</b><br>${data.textResult.replace(/\n/g, "<br>")}`;
                    addMessage(msg, "bot");
                } else if (data.error) {
                    addMessage("서버 오류: " + data.error, "bot");
                } else {
                    addMessage("인식 결과를 가져오지 못했습니다.", "bot");
                }
            } catch (err) {
                addMessage("오류 발생: " + err, "bot");
            }
        };
        reader.readAsDataURL(blob);
    });
});
