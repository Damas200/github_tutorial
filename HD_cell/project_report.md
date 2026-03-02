  
# Real-Time Customer Heartbeat Monitoring System  
## Project Report  

### Author: Damas Niyonkuru  
### Project Title: Real-Time Customer Heartbeat Monitoring System  

---

# 1. Introduction

Modern data-driven systems rely on real-time streaming architectures to process continuous event data efficiently. This project implements a complete real-time data engineering pipeline that simulates customer heart rate data, streams it via Apache Kafka, processes and validates it using a Kafka consumer, stores it in PostgreSQL, and visualizes the results through an enterprise-style Streamlit dashboard.

The primary goal of the project is to demonstrate key data engineering principles including streaming ingestion, schema validation, idempotent data processing, time-series modeling, and real-time observability.

The system simulates IoT-like heart rate monitors and processes the data end-to-end in real time.

---

# 2. System Architecture

The system follows a modular streaming architecture:

```

Synthetic Generator → Kafka Producer → Kafka Topic → Kafka Consumer → PostgreSQL → Streamlit Dashboard

```

## 2.1 Components Overview

### 1. Synthetic Data Generator
A Python script continuously generates heart rate data for multiple customers.  
Each event contains:
- customer_id  
- timestamp (UTC, ISO format)  
- heart_rate (50–140 bpm normal range)  

Anomalies are intentionally injected (<30 or >180 bpm) to simulate real-world sensor irregularities.

---

### 2. Kafka Producer
The producer serializes events as JSON and publishes them to the Kafka topic:

```

customer-heartbeat

````

Kafka ensures decoupling between data generation and data processing.

---

### 3. Kafka Consumer
The consumer:
- Subscribes to the topic
- Validates message schema
- Applies anomaly detection rules
- Inserts records into PostgreSQL
- Uses idempotent insert logic (`ON CONFLICT DO NOTHING`)

This ensures reliability and prevents duplicate records during reprocessing or restarts.

---

### 4. PostgreSQL Database
PostgreSQL is used as a structured time-series storage system.

Schema design includes:
- Primary key
- Unique constraint on (customer_id, event_timestamp)
- Indexes for efficient time-based queries
- Ingestion timestamp tracking

This supports both real-time and historical analysis.

---

### 5. Streamlit Dashboard (Observability Layer)

The dashboard provides enterprise-style monitoring with:

- Data freshness indicator
- Consumer lag estimation
- Anomaly rate tracking
- P95 heart rate percentile
- Rolling averages
- Event throughput per minute
- Customer-level summary metrics
- Data quality validation
- CSV export functionality

This simulates production monitoring dashboards used in modern data platforms.

---

# 3. Key Design Decisions

## 3.1 Use of Kafka
Kafka was selected because it:
- Supports high-throughput streaming
- Enables decoupled microservices architecture
- Provides consumer group coordination
- Ensures fault-tolerant message handling

---

## 3.2 Idempotent Consumer Design
To prevent duplicate inserts, the consumer uses:

```sql
ON CONFLICT (customer_id, event_timestamp) DO NOTHING;
````

This ensures resilience against:

* Consumer restarts
* Offset replay
* Network interruptions

Idempotency is a critical production-grade design principle.

---

## 3.3 Timezone-Aware Timestamps

All timestamps are stored in UTC and handled as timezone-aware objects to avoid distributed system inconsistencies.

This prevents tz-naive vs tz-aware bugs, which are common in real-time pipelines.

---

## 3.4 Dockerized Infrastructure

The entire infrastructure (Kafka, Zookeeper, PostgreSQL) is containerized using Docker Compose.

Benefits:

* Reproducibility
* Isolation from host conflicts
* Easy environment setup
* Production-like deployment simulation

---

# 4. Observability & Monitoring

A major enhancement of this project is the addition of an enterprise-style monitoring dashboard.

The dashboard provides:

## 4.1 Streaming Metrics

* Records processed
* Consumer lag estimation
* Event throughput per minute

## 4.2 Statistical Metrics

* Anomaly rate (%)
* P95 heart rate
* Rolling averages (10-event smoothing)

## 4.3 Data Quality Metrics

* Missing values check
* Negative heart rate detection
* Data freshness monitoring

## 4.4 Customer-Level Insights

* Average heart rate per customer
* Anomaly count per customer
* Total records per customer

These features simulate real-world streaming observability practices used in production systems.

---

# 5. Challenges Encountered & Solutions

## 5.1 Port Conflict (PostgreSQL)

A local PostgreSQL service conflicted with Docker’s port binding on 5432.

**Solution:**
Stopped local PostgreSQL service and ensured Docker owned the port.

---

## 5.2 Authentication Issues

Password mismatches occurred due to persistent Docker volumes.

**Solution:**
Reset the volume and manually updated the database user password.

---

## 5.3 Streamlit Connection Caching Error

Using cached connections caused "connection already closed" errors.

**Solution:**
Implemented safe connection handling (open → query → close per request).

---

## 5.4 Timezone Subtraction Error

Mixing timezone-aware and naive timestamps caused runtime errors.

**Solution:**
Standardized all timestamps using `datetime.now(UTC)` and UTC-aware parsing.

---

# 6. Project Outcomes

This project successfully demonstrates:

* Real-time streaming architecture
* Event-driven data ingestion
* Idempotent consumer logic
* Time-series database modeling
* Observability-driven monitoring
* Docker-based deployment
* Enterprise-grade dashboard implementation

The final system is robust, modular, and production-inspired.

---

# 7. Future Improvements

Potential enhancements include:

* Real Kafka consumer lag integration
* Machine learning anomaly detection
* Grafana-based monitoring
* Cloud deployment (AWS/GCP/Azure)
* CI/CD pipeline automation

---

# 8. Conclusion

The Real-Time Customer Heartbeat Monitoring System demonstrates a complete streaming data engineering pipeline with ingestion, validation, storage, and observability layers.

By integrating Kafka, PostgreSQL, Docker, and Streamlit, the project reflects modern data engineering best practices and production-ready architectural patterns.

This project highlights the importance of reliability, idempotency, monitoring, and structured data modeling in real-time systems.

---

