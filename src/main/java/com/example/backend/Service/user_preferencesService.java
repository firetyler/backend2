package com.example.backend.Service;

import java.util.ArrayList;
import java.util.List;


import org.springframework.stereotype.Service;

import com.example.backend.Domain.User_preferences;
import com.example.backend.Repository.user_preferencesRepository;
import com.example.backend.componets.PyhonLlmClient;
@Service
public class user_preferencesService {
    private final user_preferencesRepository user_preferences;
     private final PyhonLlmClient llm;

    public user_preferencesService(user_preferencesRepository user_preferences, PyhonLlmClient llm) {
        this.user_preferences = user_preferences;
        this.llm = llm;
    }
    public List<User_preferences> generateUser_preferences(List<String> inputs){
        try {
            List<User_preferences> dataList = new ArrayList<>();
            for(String input : inputs){
                 String response = llm.generate(input);
                 User_preferences promt = new User_preferences();
                 promt.setText(input);
                 promt.setUserInfo(response);
                 dataList.add(promt);

            }
            return user_preferences.saveAll(dataList);
        } catch (Exception e) {
            throw new RuntimeException("LLM failed: " + e.getMessage());
        }
       

    }
    public List<User_preferences> getAllUser_preferences() {
        return user_preferences.findAll();
    }
    public void saveUser_preferences(User_preferences data){
        user_preferences.save(data);

    }
}
