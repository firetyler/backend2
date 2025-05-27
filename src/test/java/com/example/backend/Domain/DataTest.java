package com.example.backend.Domain;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class DataTest {
    @Test
    void testGetId() {
        Data data = new Data();
        data.setId(123L);
        assertEquals(data, data.getId());

    }

    @Test
    void testGetInput() {
        Data data = new Data();
        data.setInput("test");
        assertEquals("test", data.getInput());

    }

    @Test
    void testGetName() {
        Data data = new Data();
        data.setName("result");
        assertEquals("result", data.getName());

    }

    @Test
    void testGetOutput() {
      Data data = new Data();
        data.setOutput("result");
        assertEquals("result", data.getOutput());

    }

    @Test
    void testSetId() {
        Data data = new Data();
        data.setId(11L);
        assertEquals(11L, data.getId());

    }

    @Test
    void testSetInput() {
        Data data = new Data();
        data.setInput("input text");
        assertEquals("input text", data.getInput());
    }

    @Test
    void testSetName() {
        Data data = new Data();
        data.setName("Bob");
        assertEquals("Bob", data.getName());

    }

    @Test
    void testSetOutput() {
        Data data = new Data();
        data.setOutput("output text");
        assertEquals("output text", data.getOutput());

    }
}
