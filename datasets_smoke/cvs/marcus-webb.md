# Marcus Webb
marcus.webb@example.com | (503) 555-0148 | Portland, OR | linkedin.com/in/marcuswebb-de | github.com/mwebb-data

## Summary
Data Engineer with 5 years of experience designing and operating batch and streaming pipelines at scale. Specializes in building reliable ELT workflows using Spark, Airflow, and dbt, with a strong focus on data quality, cost efficiency, and self-service analytics enablement. Comfortable owning systems end-to-end, from ingestion through modeled marts consumed by BI and ML teams.

## Skills
- **Orchestration:** Airflow (2.x), Dagster (basic), cron-based scheduling
- **Processing:** Apache Spark (PySpark, Spark SQL), Kafka, Flink (introductory)
- **Transformation/Modeling:** dbt Core, dbt Cloud, SQL (advanced), Jinja macros
- **Storage/Warehousing:** Snowflake, Redshift, S3, Delta Lake, Parquet
- **Languages:** Python, SQL, Bash, some Scala
- **Infra/DevOps:** Docker, Terraform, AWS (EMR, Glue, S3, Lambda), CI/CD (GitHub Actions)
- **Data Quality:** Great Expectations, dbt tests, Monte Carlo (monitoring)
- **Other:** Git, Jira, Confluence, Looker, Metabase

## Work Experience

### Senior Data Engineer — Northfield Analytics
*Jun 2023 – Present*
- Redesigned the core event-ingestion pipeline using Spark on EMR, reducing daily processing time from 6 hours to under 90 minutes by optimizing partitioning and shuffle behavior.
- Migrated 120+ legacy SQL transformation scripts into a modular dbt project, introducing testing, documentation, and lineage tracking that cut data incident response time by 40%.
- Built and maintained 30+ Airflow DAGs orchestrating ingestion, transformation, and export jobs across five business domains, with SLA monitoring and automated alerting via Slack.
- Partnered with the analytics team to design a dimensional model for marketing attribution, enabling self-service reporting that eliminated a recurring manual reporting task.

### Data Engineer — Caldwell River Logistics
*Aug 2021 – May 2023*
- Developed Spark-based ETL jobs to process 200GB+ of daily shipment and telemetry data, loading curated datasets into Redshift for operations reporting.
- Implemented incremental dbt models and snapshotting strategy for slowly changing dimensions, replacing brittle full-refresh jobs and cutting warehouse compute costs by roughly 25%.
- Authored Airflow sensors and custom operators to integrate with third-party carrier APIs, improving pipeline reliability during upstream outages.
- Introduced Great Expectations checks into the ingestion layer, catching schema drift issues before they reached downstream dashboards.

### Junior Data Engineer — Pinehollow Retail Group
*Jul 2020 – Jul 2021*
- Built Python and SQL scripts to consolidate point-of-sale data from 80+ store locations into a central Redshift warehouse on a nightly schedule.
- Assisted in migrating scheduled cron jobs to an initial Airflow deployment, documenting DAG design conventions still used by the team today.
- Wrote unit and data-quality tests for transformation logic, reducing recurring reporting discrepancies flagged by the finance team.

### Data Analyst Intern — Pinehollow Retail Group
*Jan 2020 – Jun 2020*
- Supported the analytics team by writing SQL queries and building dashboards in Metabase to track store-level sales performance.
- Automated a weekly inventory reconciliation report, saving the team an estimated 5 hours per week of manual work.

## Education
**B.S. in Computer Science**
Oregon State University — 2016 – 2020
