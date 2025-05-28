package com.example.backend.Service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;


import com.example.backend.Repository.DataRepository;
import com.example.backend.Domain.Data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.Arrays;
import java.util.List;


public class DataServiceGenerateTest {

    @Mock
    private DataRepository dataRepository;

    @InjectMocks
    private DataServiceGenerate dataServiceGenerate;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGenerateAndSave() {
    List<String> inputs = Arrays.asList("input1", "input2");

    // Mocka beteendet i repositoryn om det behövs, t.ex:
    when(dataRepository.save(any(Data.class))).thenAnswer(i -> i.getArguments()[0]);

    List<Data> savedDataList = dataServiceGenerate.generateAndSave(inputs);

    assertNotNull(savedDataList);
    assertEquals(inputs.size(), savedDataList.size());

    verify(dataRepository, times(inputs.size())).save(any(Data.class));
}

    @Test
    void testGetAllData() {
        Data d1 = new Data();
        Data d2 = new Data();
        List<Data> dataList = Arrays.asList(d1, d2);

        when(dataRepository.findAll()).thenReturn(dataList);

        List<Data> result = dataServiceGenerate.getAllData();

        assertEquals(2, result.size());
        verify(dataRepository, times(1)).findAll();
    }

    @Test
    void testSaveData() {
        Data data = new Data();
        data.setInput("test input");
        data.setOutput("test output");

        when(dataRepository.save(data)).thenReturn(data);

        Data result = dataServiceGenerate.saveData(data);

        assertEquals("test input", result.getInput());
        assertEquals("test output", result.getOutput());
        verify(dataRepository, times(1)).save(data);
    }
}
