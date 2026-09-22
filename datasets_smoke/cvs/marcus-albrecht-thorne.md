# Marcus Albrecht-Thorne
marcus.albrecht-thorne@example.com | +1 (503) 555-0148 | Portland, OR | linkedin.com/in/marcusathorne | github.com/mathorne

## Summary
Senior Site Reliability Engineer with 8 years of experience designing and operating large-scale, highly available infrastructure. Deep expertise in Kubernetes, Terraform, and Go, with a track record of reducing incident frequency, cutting cloud spend, and building internal platforms that let product teams ship faster and safer. Passionate about observability, chaos engineering, and mentoring engineers on production-readiness practices.

## Skills
- **Orchestration & Infra**: Kubernetes (EKS, GKE, self-managed), Helm, Terraform, Terragrunt, Packer, Ansible
- **Languages**: Go, Python, Bash
- **Cloud**: AWS, GCP, Azure (secondary)
- **Observability**: Prometheus, Grafana, OpenTelemetry, Datadog, PagerDuty
- **CI/CD**: ArgoCD, GitHub Actions, Jenkins, Spinnaker
- **Reliability Practices**: SLOs/SLIs, on-call rotation design, chaos engineering (Chaos Mesh, Gremlin), incident command, postmortems
- **Networking & Security**: Envoy, Istio, mTLS, VPC design, IAM policy hardening

## Work Experience

### Senior Site Reliability Engineer — Northwind Cloud Systems
*March 2022 – Present*
- Led migration of 40+ microservices from a monolithic EC2 deployment to Kubernetes (EKS), cutting infrastructure costs by 32% through right-sizing and spot-instance adoption
- Designed and rolled out a company-wide Terraform module library, reducing new-environment provisioning time from 3 days to under 2 hours
- Built a Go-based custom Kubernetes operator to automate certificate rotation across 12 clusters, eliminating manual renewal incidents
- Defined and implemented SLOs for 25 critical services, cutting P1 incidents by 45% year-over-year through proactive alerting
- Mentored 4 junior engineers through on-call rotation onboarding and production-readiness reviews

### Site Reliability Engineer — Cascadia Data Systems
*July 2019 – February 2022*
- Operated and scaled a multi-region Kubernetes platform supporting 15M+ daily active users, maintaining 99.95% uptime
- Wrote Terraform modules to standardize VPC, IAM, and RDS provisioning across 6 AWS accounts, reducing configuration drift incidents
- Developed a Go CLI tool for automated canary deployment verification, reducing rollback time from 20 minutes to under 3
- Implemented Prometheus/Grafana observability stack from scratch, replacing legacy Nagios monitoring and improving mean-time-to-detect by 60%
- Participated in and later led the on-call incident response rotation for platform-wide infrastructure

### DevOps Engineer — Fernridge Technologies
*June 2017 – June 2019*
- Automated CI/CD pipelines using Jenkins and Ansible, reducing average deployment time from 45 minutes to 8 minutes
- Containerized 20+ legacy services using Docker, laying the groundwork for the company's later Kubernetes adoption
- Built internal tooling in Python and Go for automated backup verification across PostgreSQL and Redis clusters
- Collaborated with security team to implement least-privilege IAM policies, reducing overprivileged access findings by 70%

## Education
**B.S. in Computer Science**
University of Willamette Valley, 2013 – 2017

## Certifications
- Certified Kubernetes Administrator (CKA)
- HashiCorp Certified: Terraform Associate
- AWS Certified Solutions Architect – Associate
