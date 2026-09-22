---
company: e2b
title: Platform Engineer
location: San Francisco
url: https://jobs.ashbyhq.com/e2b/c8c15520-cfd9-4b04-a902-e60a098aa96c
posted_at: 2025-09-29
---

**About E2B**
=============

E2B is a fast-growing Series A startup with over $37M in funding and 8-figure revenue. We are building the Agent Cloud and power the infrastructure behind AI labs and some of the most widely used consumer and enterprise agents, including Manus, Genspark, Lindy, Groq, and Artificial Analysis. Our team includes original authors of Firecracker, the AWS microVM technology powering Lambda, along with alumni of Cognition, JetBrains, Zapier, and Wish.

**About the role**
==================

**You will be building the next cloud platform for running AI software - a cloud where AI apps are building other software apps.**

Your job will be:

1. Building a distributed system for millions and billions of AI agents running on E2B
2. Building an orchestrator for placing sandboxes in the right nodes
3. Adding support for sandbox live migrations
4. Making sure our self-hosting DX is as smooth as possible (we’re open-source)
5. Not letting our sandboxes take more than 200ms to start (starting with the user hitting enter)
6. Scaling to millions and later billions of sandboxes running at the same time
7. Building an observability stack starting at the kernel level of virtual machines

We’re looking for an infrastructure engineer passionate about making things run fast and efficiently, and running A LOT of them at the same time.

If you aren’t afraid of going into the kernel of a VM and words like Firecracker, eBPF, UFFD, block device, L4 load balancing, noisy neighbor problem, or hugepages sound exciting to you, we want to hear from you!

**What we're looking for**
==========================

* **5+ years building distributed systems** - You've operated infrastructure at serious scale (100K+ RPS, multi-region, PB-scale data) and understand the trade-offs between consistency, availability, and partition tolerance in practice, not just theory
* **Deep Linux internals expertise** - You're comfortable working at the kernel level. You've debugged performance issues using eBPF, understand CPU scheduling, memory management, and can explain the difference between cgroups v1 and v2 without looking it up
* **VM hypervisor experience** - You've worked with Firecracker, QEMU, KVM, or similar. You understand virtio, know what a hypercall is, and have opinions about nested virtualization trade-offs
* **Systems programming skills** - Strong in at least one of: Go, Rust, C/C++. You've written performance-critical code and know when to reach for lock-free data structures, memory-mapped files, or io\_uring
* **Production orchestration experience** - You've built or operated orchestration systems (Kubernetes, Nomad, or custom). You understand bin-packing algorithms, resource scheduling, and have dealt with noisy neighbor problems at scale
* **Performance obsession** - You've shaved milliseconds off hot paths, understand CPU caches and memory locality, and have profiled production systems under load. You know what "p99 latency" means and care deeply about making it better
* **Networking expertise** - Strong understanding of L4/L7 load balancing, network namespaces, iptables/nftables, and how to build secure, isolated network topologies for multi-tenant systems
* **Located in San Francisco or willing to relocate** - We work in person as a team and believe in the magic that happens when engineers collaborate face-to-face on hard problems
* **Excited about open source** - Comfortable with our code and infrastructure being public. You contribute to discussions, write clear documentation, and help the community succeed with self-hosting

**Bonus points for:**
=====================

* Experience with userfaultfd (UFFD), copy-on-write mechanisms, or lazy loading
* GPU passthrough or PCIe device virtualization experience
* Built or maintained infrastructure for AI/ML workloads
* Contributions to Firecracker, Cloud Hypervisor, or similar open source projects
* Experience with observability at scale (distributed tracing, kernel-level metrics)

**What it’s like to work at E2B**
=================================

We’re a fast-growing startup with in-person (4 days on-site, 1 day WFH) offices in **San Francisco and Prague, Czech Republic**. We already generate 8-figure revenue and work directly with top-tier AI companies like Manus, Genspark, Lindy, Groq, Artificial Analysis and other exciting teams pushing the frontier of AI.

We offer healthcare, vision, and dental insurance, unlimited PTO, 401k, and a variety of perks for in-office employees.
