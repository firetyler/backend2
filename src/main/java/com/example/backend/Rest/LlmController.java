package com.example.backend.Rest;
import java.util.List;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.example.backend.Domain.Data;
import com.example.backend.Service.DataServiceGenerate;
import com.example.backend.Service.LlmServiceGenerate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@RestController
@RequestMapping("/api/llm")
public class LlmController {
    private final LlmServiceGenerate llmService;
    private final DataServiceGenerate dataService;

    public LlmController(LlmServiceGenerate llmService, DataServiceGenerate dataService) {
        this.llmService = llmService;
        this.dataService = dataService;
    }

  
   @PostMapping("/ask")
    public Data ask(@RequestBody PromptRequest promptRequest) {
        String input = promptRequest.getPrompt();
        String response = llmService.sendPrompt(input);

        Data data = new Data();
        data.setInput(input);
        data.setOutput(response);

        Data savedData = dataService.saveData(data); // sparar till databasen

        return savedData; // returnerar som JSON
    }

    // Skickar en lista med prompts och sparar dem i databasen

    @PostMapping("/generate")
    public List<Data> generate(@RequestBody List<String> prompts) {
        return dataService.generateAndSave(prompts);
    }

    // Hämta alla sparade prompts/respons

    @GetMapping("/history")
    public List<Data> getAll() {
        return dataService.getAllData();
    }
}