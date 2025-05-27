package com.example.backend.Service;

import org.springframework.stereotype.Service;

import com.example.backend.Domain.User;
import com.example.backend.Repository.UserRepository;
@Service
public class SetUserService {
     private final UserRepository userRepository;

    public SetUserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User saveUser(User user) {
        return userRepository.save(user);
    }

}
