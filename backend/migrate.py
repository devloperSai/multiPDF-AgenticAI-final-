"""
Refine 14 + 16 — Database migration
Adds content_hash and chroma_pdf_id columns to documents table.
Run once before starting the server.
Safe to run multiple times — uses IF NOT EXISTS.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import SessionLocal
from sqlalchemy import text

def run_migration():
    db = SessionLocal()
    try:
        # Add content_hash column (Refine 14)
        db.execute(text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)
        """))

        # Add chroma_pdf_id column (Refine 16)
        db.execute(text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS chroma_pdf_id VARCHAR(36)
        """))

        db.commit()
        print("[migration] content_hash and chroma_pdf_id columns added successfully")

    except Exception as e:
        db.rollback()
        print(f"[migration] Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    run_migration()