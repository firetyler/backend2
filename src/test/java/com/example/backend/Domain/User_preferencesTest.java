package com.example.backend.Domain;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class User_preferencesTest {
 @Test
    void testGetId() {
        User_preferences prefs = new User_preferences();
        prefs.setId(1L);
        assertEquals(1L, prefs.getId());
    }

    @Test
    void testGetName() {
        User_preferences prefs = new User_preferences();
        prefs.setName("DarkMode");
        assertEquals("DarkMode", prefs.getName());
    }

    @Test
    void testGetText() {
        User_preferences prefs = new User_preferences();
        prefs.setText("enabled");
        assertEquals("enabled", prefs.getText());
    }

    @Test
    void testGetUserInfo() {
        User_preferences prefs = new User_preferences();
        User user = new User(); // antar att du har en User-klass
        user.setUsername("s");
        prefs.setUserInfo("s");
        assertEquals(user, prefs.getUserInfo());
    }

    @Test
    void testSetId() {
        User_preferences prefs = new User_preferences();
        prefs.setId(99L);
        assertEquals(99L, prefs.getId());
    }

    @Test
    void testSetName() {
        User_preferences prefs = new User_preferences();
        prefs.setName("Language");
        assertEquals("Language", prefs.getName());
    }

    @Test
    void testSetText() {
        User_preferences prefs = new User_preferences();
        prefs.setText("English");
        assertEquals("English", prefs.getText());
    }

    @Test
    void testSetUserInfo() {
        User_preferences prefs = new User_preferences();
        User user = new User();
        user.setUsername("tester");
        prefs.setUserInfo("s");
        assertEquals("tester", prefs.getUserInfo(),user.getUsername());
    }
}
