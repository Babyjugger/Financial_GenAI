# config.py
EMAIL_CONFIG = {
    "enabled": True,
    "recipient": "babyjugger@live.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "babyjonathanlim@gmail.com",
    "smtp_password": "uefw likm ywvs tqvg"  # Consider using environment variables
}

PDF_CONFIG = {
    "page_size": "letter",
    "margins": "2cm",
    "include_header": True,
    "include_footer": True
}

CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "./cache",
    "expire_days": 30
}