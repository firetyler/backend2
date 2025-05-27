package com.example.backend.Domain;
import static org.assertj.core.api.Assertions.assertThat;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

public class UserTest {

    @Test
    void testGetId() {
        User user = new User();
        user.setId(42L);
        assertThat(user.getId()).isEqualTo(42L);
    }

    @Test
    void testGetUsername() {
        User user = new User();
        user.setUsername("testuser");
        assertThat(user.getUsername()).isEqualTo("testuser");
    }

    @Test
    void testGetPreferencesList() {
        User user = new User();
        List<User_preferences> prefs = new ArrayList<>();
        user.setPreferencesList(prefs);
        assertThat(user.getPreferencesList()).isSameAs(prefs);
    }

    @Test
    void testSetId() {
        User user = new User();
        user.setId(10L);
        assertThat(user.getId()).isEqualTo(10L);
    }

    @Test
    void testSetUsername() {
        User user = new User();
        user.setUsername("newuser");
        assertThat(user.getUsername()).isEqualTo("newuser");
    }

    @Test
    void testSetPreferencesList() {
        User user = new User();
        List<User_preferences> prefs = new ArrayList<>();
        User_preferences pref = new User_preferences();
        pref.setName("pref1");
        prefs.add(pref);
        user.setPreferencesList(prefs);
        assertThat(user.getPreferencesList()).contains(pref);
    }
}

