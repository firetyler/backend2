package com.example.backend.Service;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Arrays;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.example.backend.Domain.User_preferences;
import com.example.backend.Repository.user_preferencesRepository;

public class user_preferencesServiceTest {

    @Mock
    private user_preferencesRepository repository;

    @InjectMocks
    private user_preferencesService service;  // Observera att detta är service-klassen

    @BeforeEach
    void setup() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGenerateUser_preferences() {
        List<String> prompts = Arrays.asList("prompt1", "prompt2");

        List<User_preferences> prefs = service.generateUser_preferences(prompts);

        assertEquals(2, prefs.size());
        assertEquals("prompt1", prefs.get(0).getName());
        assertEquals("prompt2", prefs.get(1).getName());
    }

    @Test
    void testGetAllUser_preferences() {
        User_preferences pref1 = new User_preferences();
        pref1.setName("pref1");
        User_preferences pref2 = new User_preferences();
        pref2.setName("pref2");

        when(repository.findAll()).thenReturn(Arrays.asList(pref1, pref2));

        List<User_preferences> allPrefs = service.getAllUser_preferences();

        assertEquals(2, allPrefs.size());
        verify(repository, times(1)).findAll();
    }

    @Test
    void testSaveUser_preferences() {
        User_preferences pref = new User_preferences();
        pref.setName("pref");

        when(repository.save(pref)).thenReturn(pref);

        User_preferences savedPref = service.saveUser_preferences(pref);

        assertEquals("pref", savedPref.getName());
        verify(repository, times(1)).save(pref);
    }
}
