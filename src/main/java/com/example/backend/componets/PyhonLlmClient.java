package com.example.backend.componets;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import org.springframework.stereotype.Component;

@Component
public class PyhonLlmClient {
    private final HttpClient client = HttpClient.newHttpClient();
    public String generate(String promt) throws IOException, InterruptedException{
        String json = "{\"prompt\": \"%s\"}".formatted(promt.replace("\"", "\\\""));
        HttpRequest request = HttpRequest.newBuilder().header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(json)).build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();

    }

}
