package com.example.backend.Service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

public class LlmServiceGenerateTest {

    @Mock
    private RestTemplate restTemplate;

    @InjectMocks
    private LlmServiceGenerate llmServiceGenerate;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testSendPrompt_success() {
        String prompt = "Hello AI";
        String expectedResponse = "{\"response\": \"Hi there!\"}";

        ResponseEntity<String> mockResponse = new ResponseEntity<>(expectedResponse, HttpStatus.OK);

        when(restTemplate.postForEntity(anyString(), any(), eq(String.class))).thenReturn(mockResponse);

        String actualResponse = llmServiceGenerate.sendPrompt(prompt);

        assertEquals(expectedResponse, actualResponse);

        verify(restTemplate, times(1)).postForEntity(anyString(), any(), eq(String.class));
    }

    @Test
    void testSendPrompt_failure() {
        String prompt = "Hello AI";

        when(restTemplate.postForEntity(anyString(), any(), eq(String.class))).thenThrow(new RuntimeException("Connection error"));

        String actualResponse = llmServiceGenerate.sendPrompt(prompt);

        assertEquals("Error: Could not connect to Flask.", actualResponse);

        verify(restTemplate, times(1)).postForEntity(anyString(), any(), eq(String.class));
    }
}
