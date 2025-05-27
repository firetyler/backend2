package com.example.backend.Domain;

import java.util.ArrayList;
import java.util.List;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
@Table(name = "app_user")  // Undvik reserverade ord som "user"
@Entity
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(length = 100)
    private String username;

    @OneToMany(cascade = CascadeType.ALL, mappedBy = "user")
    private List<User_preferences> preferencesList = new ArrayList<>();
    // Getters och setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
   public List<User_preferences> getPreferencesList() { return preferencesList; }
    public void setPreferencesList(List<User_preferences> preferencesList) { this.preferencesList = preferencesList; }

}
