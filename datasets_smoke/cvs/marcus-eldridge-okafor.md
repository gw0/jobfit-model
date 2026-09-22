# Marcus Eldridge-Okafor
marcus.eldridge-okafor@example.com | +1 (503) 555-0148 | Portland, OR | linkedin.com/in/marcuseldridgeokafor

## Summary
Senior Site Reliability Engineer with 8 years of experience designing and operating large-scale, highly available infrastructure. Deep expertise in Kubernetes, Terraform, and Go, with a track record of reducing incident volume, cutting infrastructure spend, and building tooling that lets product teams ship faster without sacrificing reliability. Comfortable owning systems end-to-end, from on-call response through long-term architectural planning.

## Skills
- **Orchestration & Infrastructure:** Kubernetes (EKS, GKE, self-managed), Helm, Terraform, Packer, Ansible
- **Languages:** Go, Python, Bash
- **Observability:** Prometheus, Grafana, OpenTelemetry, Datadog, PagerDuty
- **Cloud Platforms:** AWS, GCP
- **CI/CD:** GitHub Actions, ArgoCD, Jenkins, Spinnaker
- **Practices:** SLO/SLA design, chaos engineering, capacity planning, incident command, on-call leadership

## Work Experience

### Staff/Senior SRE — Northwind Cloud Systems
*Mar 2022 – Present*
- Led migration of 40+ microservices from a monolithic ECS deployment to a multi-region Kubernetes platform, cutting deployment time from 45 minutes to under 5.
- Designed and rolled out a Terraform module library adopted by 12 engineering teams, reducing new-environment provisioning time from 2 weeks to 1 day.
- Built a Go-based auto-remediation service that resolves the top 5 recurring alert classes automatically, reducing on-call pages by 63%.
- Established SLOs for all customer-facing services and drove error-budget-based release gating, cutting customer-impacting incidents by 35% year over year.
- Mentored 4 junior engineers and ran a quarterly incident-review program that reduced mean time to resolution (MTTR) by 40%.

### Site Reliability Engineer — Alderbrook Systems
*Jul 2019 – Feb 2022*
- Owned the on-call rotation and incident response process for a 200+ node Kubernetes fleet serving 15M+ daily requests.
- Wrote a custom Go operator to automate certificate rotation and secrets management across clusters, eliminating a class of expiry-related outages entirely.
- Migrated infrastructure-as-code from hand-rolled CloudFormation scripts to Terraform, reducing configuration drift incidents by 80%.
- Implemented cluster autoscaling policies that cut compute costs by 28% while maintaining p99 latency targets during peak traffic.

### DevOps Engineer — Fenwick Data Labs
*Jun 2017 – Jun 2019*
- Containerized legacy Java and Python services and deployed them onto a new Kubernetes cluster, improving deployment frequency from monthly to weekly.
- Built CI/CD pipelines in Jenkins and later GitHub Actions, reducing average build-to-deploy time by 70%.
- Set up centralized logging and metrics pipelines (ELK, Prometheus/Grafana) that gave the engineering org its first unified view of production health.
- Automated routine database backup and failover testing using Python and Bash, reducing manual ops toil by roughly 10 hours per week.

## Education
**B.S. in Computer Science**
Cascadia State University, 2013 – 2017

## Certifications
- Certified Kubernetes Administrator (CKA)
- HashiCorp Certified: Terraform Associate
- AWS Certified Solutions Architect – Associate
