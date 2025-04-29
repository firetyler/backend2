package com.example.backend.Rest;

import java.io.IOException;

import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.backend.componets.PyhonLlmClient;
import org.springframework.web.bind.annotation.PostMapping;

@RestController
@RequestMapping("/api/llm")
public class LlmController {
    private final PyhonLlmClient llmClient;


    public LlmController(PyhonLlmClient llmClient){
        this.llmClient = llmClient;
    }
    @PostMapping("/generate")
    public String generate(@RequestBody String promt) throws IOException, InterruptedException {
        return llmClient.generate(promt);
    }
    
}
