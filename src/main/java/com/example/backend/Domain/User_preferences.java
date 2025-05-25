package com.example.backend.Domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class User_preferences {

    @Id
    @GeneratedValue (strategy  = GenerationType.IDENTITY)
    private long id;

    private String name ;
    @Column(length = 100000)
    private String text;
    @Column(length = 100000)
    private String userInfo;
    public long getId() {
        return id;
    }
    public void setId(long id) {
        this.id = id;
    }
    public String getText() {
        return text;
    }
    public void setText(String text) {
        this.text = text;
    }
    public String getUserInfo() {
        return userInfo;
    }
    public void setUserInfo(String userInfo) {
        this.userInfo = userInfo;
    }
    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
}
