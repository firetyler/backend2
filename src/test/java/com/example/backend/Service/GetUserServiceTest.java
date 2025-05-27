package com.example.backend.Service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.example.backend.Domain.User;
import com.example.backend.Repository.UserRepository;
import com.example.backend.Service.GetUserService;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.Arrays;
import java.util.List;

public class GetUserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private GetUserService getUserService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGenerateUser() {
        List<String> prompts = Arrays.asList("alice", "bob");

        List<User> users = getUserService.generateUser(prompts);

        assertEquals(2, users.size());
        assertEquals("alice", users.get(0).getUsername());
        assertEquals("bob", users.get(1).getUsername());
    }

    @Test
    void testGetAllUser() {
        User user1 = new User();
        user1.setUsername("alice");

        User user2 = new User();
        user2.setUsername("bob");

        when(userRepository.findAll()).thenReturn(Arrays.asList(user1, user2));

        List<User> users = getUserService.getAllUser();

        assertEquals(2, users.size());
        assertEquals("alice", users.get(0).getUsername());
        assertEquals("bob", users.get(1).getUsername());

        verify(userRepository, times(1)).findAll();
    }
}
