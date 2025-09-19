package com.example.GlucoSnap.Controller;

import com.example.GlucoSnap.FastApiRunner;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/fastapi")
public class FastApiController {

    private final FastApiRunner fastApiRunner;

    public FastApiController(FastApiRunner fastApiRunner) {
        this.fastApiRunner = fastApiRunner;
    }

    @PostMapping("/start/{mode}")
    public String startServer(@PathVariable int mode) {
        try {
            fastApiRunner.startServer(mode);
            return "서버 실행됨: " + mode;
        } catch (Exception e) {
            return "실행 오류: " + e.getMessage();
        }
    }

    @PostMapping("/stop")
    public String stopServer() {
        fastApiRunner.stopServer();
        return "서버 종료됨";
    }

}
