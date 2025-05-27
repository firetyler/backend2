package com.example.backend.Service;

import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;

import com.example.backend.Domain.User;
import com.example.backend.Repository.UserRepository;

@Service
public class GetUserService {
    private final UserRepository userRepository;

    
    public GetUserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public List<User> generateUser(List<String> prompts) {
        List <User> users = new ArrayList<>();
        for (String prompt : prompts) {
            User user = new User();
            user.setUsername(prompt);
            users.add(user);
            
        }
       return users;
    }

	public List<User> getAllUser() {
		
		return userRepository.findAll();
	}

}
