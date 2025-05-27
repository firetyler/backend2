package com.example.backend.Service;

import java.util.ArrayList;
import java.util.List;


import org.springframework.stereotype.Service;

import com.example.backend.Domain.User_preferences;
import com.example.backend.Repository.user_preferencesRepository;

@Service
public class user_preferencesService {
    private final user_preferencesRepository user_preferences;
    

    public user_preferencesService(user_preferencesRepository user_preferences) {
        this.user_preferences = user_preferences;
        
    }
  public List<User_preferences> generateUser_preferences(List<String> prompts) {
        List<User_preferences> prefs = new ArrayList<>();
        for (String prompt : prompts) {
            User_preferences pref = new User_preferences();
            pref.setName(prompt);
            prefs.add(pref);
        }
        return prefs;
    }
    public List<User_preferences> getAllUser_preferences() {
        return user_preferences.findAll();
    }
    public User_preferences  saveUser_preferences(User_preferences pref){
        return user_preferences.save(pref);

    }
}
