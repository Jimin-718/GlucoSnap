package com.example.GlucoSnap;

import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

@Service
public class FastApiRunner {
    private Process currentProcess; // 실행 중인 서버 프로세스 저장

    // 서버 실행
    public void startServer(int mode) throws IOException {
        // 실행 중인 서버가 있다면 먼저 종료
        stopServer();

        ProcessBuilder pb;
        if (mode == 1) { // 사진인식 서버
            pb = new ProcessBuilder(
                    "python", "gsimagepredictFastAPI.py"
            );
        } else { // 그림인식 서버
            pb = new ProcessBuilder(
                    "python", "quickdrawFastAPI.py"
            );
        }

        pb.directory(new File("C:\\mbc12AI\\spring_boot\\GlucoSnap\\python_scripts"));
        pb.inheritIO();

        currentProcess = pb.start();
        System.out.println("FastAPI 서버 실행됨: mode=" + mode);
    }

    // 서버 종료
    public void stopServer() {
        if (currentProcess != null && currentProcess.isAlive()) {
            currentProcess.destroy();
            System.out.println("FastAPI 서버 종료됨");
        }
    }
}
