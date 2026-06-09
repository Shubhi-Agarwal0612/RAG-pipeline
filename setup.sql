-- Run this script to set up the database:
-- sudo -u postgres psql -f setup.sql

CREATE DATABASE rag_app;
\c rag_app

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_size INTEGER,
    file_type VARCHAR(50),
    file_path VARCHAR(500) NOT NULL,
    status VARCHAR(20) DEFAULT 'uploaded',
    date_uploaded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id),
    page_no INTEGER,
    text TEXT NOT NULL,
    vector vector(384)
);
