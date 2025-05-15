package com.example.backend.Service;
import java.util.HashMap;
import java.util.Map;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class LlmServiceGenerate {

    private final RestTemplate restTemplate = new RestTemplate();

    public String sendPrompt(String prompt) {
        // Prepare headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Prepare request body
        Map<String, String> body = new HashMap<>();
        body.put("prompt", prompt);

        // Wrap into HttpEntity
        HttpEntity<Map<String, String>> requestEntity = new HttpEntity<>(body, headers);

        // Send POST request to Flask
        String flaskUrl = "http://localhost:5000/generate";

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(flaskUrl, requestEntity, String.class);
            return response.getBody(); // Flask's response
        } catch (Exception e) {
            e.printStackTrace();
            return "Error: Could not connect to Flask.";
        }
    }
}

