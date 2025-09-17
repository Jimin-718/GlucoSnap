package com.example.GlucoSnap.Controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/python")
@Slf4j
public class PythonController {

    private final ObjectMapper objectMapper = new ObjectMapper();

    // 상대 경로
    @Value("${python.script.path:python_scripts/gsimagepredictFastAPI.py}")
    private String pythonScriptPath;

    @PostMapping("/predict")
    public ResponseEntity<?> predictFood(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "currentGlucose", required = false) Double currentGlucose
    ) {
        if (file.isEmpty()) return ResponseEntity.badRequest().body("파일이 업로드되지 않았습니다.");

        try {
            // 임시 저장
            String uploadDir = System.getProperty("java.io.tmpdir");
            File savedFile = new File(uploadDir, file.getOriginalFilename());
            file.transferTo(savedFile);

            // 파이썬 실행 (상대 경로 사용)
            ProcessBuilder pb = new ProcessBuilder(
                    "python", pythonScriptPath,
                    "--image", savedFile.getAbsolutePath(),
                    "--current", currentGlucose != null ? currentGlucose.toString() : ""
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // 출력 읽기
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder outputText = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                outputText.append(line);
            }
            process.waitFor();

            // 파이썬에서 JSON 출력 가정
            Map<String, Object> result = objectMapper.readValue(outputText.toString(), Map.class);

            return ResponseEntity.ok(result);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("서버 오류: " + e.getMessage());
        }
    }

    @GetMapping("/curve")
    public ResponseEntity<byte[]> getCurveImage(@RequestParam String path) {
        try {
            // TEMP 폴더 경로
            String tempDir = System.getenv("TEMP");
            File imgFile = new File(tempDir, path);

            if (!imgFile.exists()) {
                return ResponseEntity.notFound().build();
            }

            // 한글 경로 안전하게 URL 인코딩
            byte[] imageBytes = java.nio.file.Files.readAllBytes(imgFile.toPath());

            return ResponseEntity.ok()
                    .header("Content-Disposition", "inline; filename=\"" + java.net.URLEncoder.encode(imgFile.getName(), "UTF-8") + "\"")
                    .contentType(MediaType.IMAGE_PNG)
                    .body(imageBytes);

        } catch (IOException e) {
            return ResponseEntity.status(500).build();
        }
    }
}
