package com.example.backend.Service;

import org.springframework.stereotype.Service;

import com.example.backend.componets.PyhonLlmClient;

@Service
public class LlmService {
    private final PyhonLlmClient llmClient;

    public LlmService(PyhonLlmClient llmClient) {
        this.llmClient = llmClient;
    }
    public String sendPromt(String promt){
        try {
            return llmClient.generate(promt);
        } catch (Exception e) {
           return "Error communicating with LLM: " + e.getMessage();
        }
    }

}
