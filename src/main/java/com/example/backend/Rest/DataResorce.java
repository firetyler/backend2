package com.example.backend.Rest;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.example.backend.Domain.Data;
import com.example.backend.Service.DataServiceGenerate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import java.util.List;

@RestController
@RequestMapping("/api/promts")
public class DataResorce {
    private final DataServiceGenerate service;

    public DataResorce(DataServiceGenerate service) {
        this.service = service;
    }

    @PostMapping
    public List<Data> generateData(@RequestBody List<String> prompts) {
        // Ta emot en lista av strängar och generera data för alla
        return service.generateAndSave(prompts);
    }
     @GetMapping
    public List<Data> getAllPromts() {
        return service.getAllData(); 
    }
}
