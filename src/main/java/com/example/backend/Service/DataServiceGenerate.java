package com.example.backend.Service;
import org.springframework.stereotype.Service;
import com.example.backend.Domain.Data;
import com.example.backend.Repository.DataRepository;
import com.example.backend.componets.PyhonLlmClient;
import java.util.ArrayList;
import java.util.List;

@Service
public class DataServiceGenerate {
    private final DataRepository dataRepository;
    private final PyhonLlmClient llm;


    public DataServiceGenerate(DataRepository dataRepository, PyhonLlmClient llm) {
        this.dataRepository = dataRepository;
        this.llm = llm;
    }

    public List<Data> generateAndSave(List<String> inputs) {
        try {
            // Skapa en lista för att hålla alla Data-objekt
            List<Data> dataList = new ArrayList<>();

            // Loopa genom varje input-sträng
            for (String input : inputs) {
                // Generera svar från LLM
                String response = llm.generate(input);

                // Skapa ett nytt Data-objekt
                Data prompt = new Data();
                prompt.setInput(input);
                prompt.setOutput(response);

                // Lägg till objektet i listan
                dataList.add(prompt);
            }

            // Spara alla Data-objekt i databasen
            return dataRepository.saveAll(dataList); // SaveAll sparar alla objekt i databasen
        } catch (Exception e) {
            throw new RuntimeException("LLM failed: " + e.getMessage());
        }
    }
    public List<Data> getAllData() {
        return dataRepository.findAll();
    }
    public void saveData(Data data) {
        dataRepository.save(data);  // Här sparar vi objektet till databasen
    }
}

