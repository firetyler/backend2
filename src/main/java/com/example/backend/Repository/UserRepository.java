package com.example.backend.Repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.example.backend.Domain.User;

public interface UserRepository extends JpaRepository<User , Long> {

}
