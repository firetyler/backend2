package com.example.backend.Rest;


import com.example.backend.Service.DataServiceGenerate;
import com.example.backend.Domain.Data;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(DataResorce.class)
public class DataResorceTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private DataServiceGenerate dataService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void testGenerateData() throws Exception {
        Data dummy = new Data();
        dummy.setName("test");
        dummy.setInput("Hello");
        dummy.setOutput("Hi there!");

        when(dataService.generateAndSave(any())).thenReturn(List.of(dummy));

        mockMvc.perform(post("/api/data/generate")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(List.of("Hello"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].output").value("Hi there!"));
    }
}
