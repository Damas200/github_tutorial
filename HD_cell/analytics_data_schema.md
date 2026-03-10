# QuickPoll Analytics Data Schema

## Overview

This document describes the analytics datasets produced by the **QuickPoll Data Engineering ETL pipeline**.

The pipeline aggregates operational application data into analytics-ready tables that support:

- product analytics
- dashboard visualizations
- backend reporting APIs
- user engagement insights

The ETL pipeline is implemented in:

```
data-engineering/etl_pipeline.py
```

These analytics tables are **read-only datasets** designed for analytics and reporting.

---

# Source Data

The ETL pipeline extracts data from the following application tables:

| Table | Description |
|------|-------------|
| users | Application users |
| polls | Polls created by users |
| poll_options | Options available for polls |
| votes | Votes cast by users |

---

# Analytics Tables

## analytics_poll_summary

Aggregated statistics for each poll.

| Column | Type | Description |
|------|------|-------------|
| id | integer | Poll identifier |
| question | text | Poll question |
| creator_name | text | Name of poll creator |
| total_votes | integer | Total votes received |
| created_at | timestamp | Poll creation timestamp |
| etl_run_at | timestamp | ETL execution timestamp |

---

## analytics_vote_trends

Daily voting activity across the platform.

| Column | Type | Description |
|------|------|-------------|
| vote_date | date | Date votes were cast |
| votes_per_day | integer | Number of votes recorded on that day |
| etl_run_at | timestamp | ETL execution timestamp |

---

## analytics_user_participation

User engagement metrics based on voting activity.

| Column | Type | Description |
|------|------|-------------|
| voter_name | text | User who cast the vote |
| total_votes_cast | integer | Total votes cast by the user |
| participation_rate | float | Calculated user participation metric |
| etl_run_at | timestamp | ETL execution timestamp |

---

## etl_pipeline_runs

Metadata table used to track ETL pipeline executions.

| Column | Type | Description |
|------|------|-------------|
| id | integer | Pipeline run identifier |
| pipeline_name | text | Name of the ETL pipeline |
| run_started_at | timestamp | Pipeline start time |
| run_finished_at | timestamp | Pipeline finish time |
| status | text | Pipeline execution status |
| rows_processed | integer | Number of processed rows |
| duration_seconds | float | Pipeline runtime duration |

---

# Data Pipeline Flow

Application tables are transformed into analytics datasets through the ETL pipeline:

Application Tables  
(users, polls, votes)

↓

ETL Pipeline

↓

Analytics Tables

---

# Data Freshness

Each analytics table includes an **etl_run_at** column that records when the ETL pipeline generated the dataset.

---

# Intended Consumers

| Team | Usage |
|------|------|
| Backend | Analytics API endpoints |
| Frontend | Dashboard visualizations |
| Product | Engagement analysis |
| QA | Data validation |
| DevOps | Pipeline monitoring |


