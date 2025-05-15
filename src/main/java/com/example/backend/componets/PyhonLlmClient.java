package com.example.backend.componets;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import org.springframework.stereotype.Component;

@Component
public class PyhonLlmClient {
    private final HttpClient client = HttpClient.newHttpClient();

    public String generate(String prompt) throws IOException, InterruptedException {
        String json = "{\"prompt\": \"%s\"}".formatted(prompt.replace("\"", "\\\""));
    
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:5000/generate")) // <-- RÄTT PORT
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build();
    
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public String getHistory() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:5000/history")) // Correct URL for history
                .header("Content-Type", "application/json")
                .GET() // Use GET for retrieving data
                .build();
    
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public String askFlaskLLM(String prompt) throws IOException, InterruptedException {
    String jsonBody = "{ \"prompt\": \"" + prompt + "\" }";

    HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:5000/ask"))  // OBS: endpointen är /ask
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();

    HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

    System.out.println("Response code: " + response.statusCode());
    System.out.println("Response body: " + response.body());
    return response.body();
}
    
}