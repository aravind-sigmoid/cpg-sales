"""Create all tables. Run once before first ingestion."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import engine, Base
import db.models  # noqa: F401 — import models so Base knows about them


def init_db() -> None:
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


if __name__ == "__main__":
    init_db()
