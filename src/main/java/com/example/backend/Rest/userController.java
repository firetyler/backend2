package com.example.backend.Rest;

import java.util.List;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.backend.Domain.User;
import com.example.backend.Service.GetUserService;


@RestController
@RequestMapping("/api/user")
public class userController {
    private final GetUserService service;

    public userController(GetUserService service) {
        this.service = service;
    }
    @PostMapping
    public List<User> generateUser(@RequestBody List<String> prompts){
        return service.generateUser(prompts);
    }

    public List<User> getAllUsers (){
        return service.getAllUser();

    }

}
