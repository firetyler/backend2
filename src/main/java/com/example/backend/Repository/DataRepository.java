package com.example.backend.Repository;

import com.example.backend.Domain.Data;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DataRepository extends JpaRepository<Data, String> {
}
