# OmniClip SaaS

OmniClip is an elite AI-powered Video and Carousel generation SaaS platform designed to transform raw scripts and prompts into highly engaging, production-ready short-form content. Using automated asset retrieval, AI timeline scheduling, and localized FFmpeg micro-rendering pipelines, OmniClip delivers optimized media packaging at scale.

---

## 🛠️ Technology Stack

OmniClip is built on a high-concurrency, security-first architecture utilizing:
- **Backend Framework:** Django (REST Framework)
- **Task Queue & Scheduler:** Celery & Redis
- **Database:** PostgreSQL (with AES-256 field encryption for tokens and API keys)
- **Video Processing Engine:** FFmpeg & Pillow
- **Frontend Application:** Next.js (React Dashboard, Zustand state management)

---

## 📄 Core Project Documentation

This repository maintains strict specifications and system guardrails. Refer to the files below in the `docs/` folder to understand the engineering design, schema boundaries, and architectural guidelines:

| Document | Description |
| :--- | :--- |
| 🛡️ [docs/01_STRICT_RULES.md](docs/01_STRICT_RULES.md.txt) | **System Architecture & Guardrails:** Core safety regulations regarding concurrency, rate throttling, AES-256, auto-refunds, and offline media fallbacks. |
| 🗄️ [docs/02_DATABASE_SCHEMA.md](docs/02_DATABASE_SCHEMA.md.txt) | **PostgreSQL Core Tables:** Table configurations including User tier mappings, Credits economy, Project timelines, Project statuses, and BYOK credentials. |
| 🔌 [docs/03_BACKEND_API.md](docs/03_BACKEND_API.md.txt) | **Django REST Endpoints:** Detailed documentation on trend scrapers, script generation, full/chunk rendering requests, and social schedule posting. |
| ⚙️ [docs/04_AUTOMATION_PIPELINE.md](docs/04_AUTOMATION_PIPELINE.md.txt) | **Celery Flow & Media Sourcing:** Deep-dive into background tasks, video assembly, subtitle burning, multi-track audio ducking, and API retry behaviors. |
| 🎨 [docs/05_FRONTEND_UI.md](docs/05_FRONTEND_UI.md.txt) | **React Dashboard & Zustand:** Frontend state diagrams, component flows, timeline video player hooks, and optimized local REST patches. |
| 📂 [docs/06_PROJECT_STRUCTURE.md](docs/06_PROJECT_STRUCTURE.md) | **Folder Architecture:** Complete file layout guidelines and strict gitignore settings for secrets, media, and environments. |

---

## 🤖 AI Agent Guidelines & Strict Guardrails

> [!IMPORTANT]
> **Strict Guardrails for Coding Assistants**
> All developer agents, subagents, and LLMs operating on this codebase must strictly read and adhere to the [.cursorrules](.cursorrules) file located at the root of the project before proposing or executing any code edits.
>
> **Core Developer Rules:**
> 1. **No Assumptions:** Never modify the database models or views without cross-referencing files in the `docs/` folder.
> 2. **AES-256 Encryption:** Secure all external API credentials (Gemini, ElevenLabs, Pexels) and OAuth social tokens using AES-256 (`cryptography.fernet`) before database persistence. Never log or output plain-text secrets.
> 3. **Isolated Tasks**: CPU-intensive operations (such as FFmpeg video processing and audio ducking) must execute strictly inside Celery workers, never directly within the request-response thread of Django views.
