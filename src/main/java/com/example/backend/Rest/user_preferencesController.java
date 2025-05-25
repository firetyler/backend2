package com.example.backend.Rest;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


import com.example.backend.Domain.User_preferences;
import com.example.backend.Service.user_preferencesService;

@RestController
@RequestMapping("/api/user_preferences")
public class user_preferencesController {
    private final user_preferencesService service;

    public user_preferencesController(user_preferencesService service) {
        this.service = service;
    }
    @PostMapping
    public List<User_preferences> generateUser_preferences(@RequestBody List<String> prompts) {
        // Ta emot en lista av strängar och generera data för alla
        return service.generateUser_preferences(prompts);
    }
     @GetMapping
    public List<User_preferences> getAllUser_preferences() {
        return service.getAllUser_preferences(); 
    }

}
