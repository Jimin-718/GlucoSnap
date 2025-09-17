package com.example.GlucoSnap.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class FrontController {

    @GetMapping(value = "/")
    public String k1(){
        return "chat";
    }

    @GetMapping(value = "/home")
    public String k2(){
        return "chat";
    }

}
