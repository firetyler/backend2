package com.example.backend.Rest;

import com.example.backend.Service.LlmServiceGenerate;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(LlmController.class)
public class LlmControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private LlmServiceGenerate llmService;

    @Test
    void testAsk() throws Exception {
        String input = "Vad är AI?";
        String response = "AI står för artificiell intelligens.";

        when(llmService.sendPrompt(input)).thenReturn(response);

        mockMvc.perform(MockMvcRequestBuilders.get("/api/llm/ask")
                .param("question", input))
                .andExpect(status().isOk())
                .andExpect(content().string(response));
    }

    @Test
    void testGenerate() throws Exception {
        String prompt = "Skriv en dikt om våren";
        String generatedText = "Våren spirar, knoppar brister...";

        when(llmService.sendPrompt(prompt)).thenReturn(generatedText);

        mockMvc.perform(MockMvcRequestBuilders.post("/api/llm/generate")
                .contentType(MediaType.TEXT_PLAIN)
                .content(prompt))
                .andExpect(status().isOk())
                .andExpect(content().string(generatedText));
    }

    @Test
    void testGetAll() throws Exception {
        // Antag att getAll returnerar en lista av tidigare inputs eller liknande
        when(llmService.getAll()).thenReturn(java.util.List.of("Fråga 1", "Fråga 2"));

        mockMvc.perform(MockMvcRequestBuilders.get("/api/llm/history"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0]").value("Fråga 1"))
                .andExpect(jsonPath("$[1]").value("Fråga 2"));
    }
}
