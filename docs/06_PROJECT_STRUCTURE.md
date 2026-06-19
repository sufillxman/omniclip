# 06_PROJECT_STRUCTURE.md (Django Industry Setup)

## Folder Structure
omniclip_saas/
├── .venv/                      # Python Virtual Environment (Strictly Local)
├── .env                        # Environment Variables (DO NOT SHARE)
├── .gitignore                  # Prevents secrets/media from leaking to GitHub
├── requirements.txt            # Python dependencies
├── manage.py
├── omniclip_core/              # Main Settings
│   ├── settings.py
│   ├── celery.py               # Celery Configuration
│   └── urls.py
├── apps/                       # Modular Django Apps
│   ├── accounts/               # BYOK, Brand Kits, Stripe, Users
│   ├── engine/                 # SEO Scraper, Gemini Prompts, Timeline Logic
│   ├── processor/              # FFmpeg Chunking, Pillow, Celery Tasks
│   └── publisher/              # AES Decryption, OAuth, Social APIs
└── media/
    ├── fallbacks/
    ├── chunks_temp/            
    └── final_outputs/

## The Strict `.gitignore`
__pycache__/
*.py[cod]
.venv/
.env
db.sqlite3
media/
!media/fallbacks/