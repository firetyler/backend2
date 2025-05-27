package com.example.backend.Service;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import com.example.backend.Domain.User;
import com.example.backend.Repository.UserRepository;


import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import org.junit.jupiter.api.extension.ExtendWith;

@ExtendWith(MockitoExtension.class)
public class SetUserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private SetUserService setUserService;

    @Test
    void testSaveUser() {
        User user = new User();
        user.setUsername("testuser");

        when(userRepository.save(user)).thenReturn(user);

        User savedUser = setUserService.saveUser(user);

        assertNotNull(savedUser);
        assertEquals("testuser", savedUser.getUsername());
        verify(userRepository, times(1)).save(user);
    }
}
