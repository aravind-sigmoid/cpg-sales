"""
Generate synthetic CPG data with realistic quality issues.

Produces three CSV files in the data/ directory:
  - raw_transactions.csv     (sales events with injected quality issues)
  - product_catalog.csv      (clean product master)
  - store_regions.csv        (store reference table)

Run: python data/generate_data.py
"""
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Reference data ────────────────────────────────────────────────────────────

CATEGORIES = ["Beverages", "Snacks", "Dairy", "Personal Care", "Household"]
BRANDS = {
    "Beverages":     ["HydraFlow", "ZestDrink", "PureSip"],
    "Snacks":        ["CrunchCo", "MunchBite", "SnackPal"],
    "Dairy":         ["FreshMoo", "CreamyGood", "YoguFarm"],
    "Personal Care": ["GlowUp", "CleanBreeze", "SoftTouch"],
    "Household":     ["SparkleHome", "CleanWave", "TidyUp"],
}
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
STATES_BY_REGION = {
    "Northeast":  ["NY", "MA", "CT", "NJ", "PA"],
    "Southeast":  ["FL", "GA", "NC", "SC", "VA"],
    "Midwest":    ["IL", "OH", "MI", "IN", "WI"],
    "West":       ["CA", "WA", "OR", "CO", "AZ"],
    "Southwest":  ["TX", "NM", "NV", "UT", "OK"],
}
DEMOGRAPHICS = ["urban", "suburban", "rural"]
SOURCE_SYSTEMS = ["POS_NORTH", "POS_SOUTH", "ONLINE", "POS_WEST"]

NUM_SKUS = 50
NUM_STORES = 40
NUM_TRANSACTIONS = 30000   # larger dataset so monthly aggregates have ~900 rows
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)


# ── Product catalog ───────────────────────────────────────────────────────────

def generate_product_catalog() -> pd.DataFrame:
    rows = []
    for i in range(NUM_SKUS):
        cat = random.choice(CATEGORIES)
        brand = random.choice(BRANDS[cat])
        rows.append({
            "sku": f"SKU-{i+1:04d}",
            "product_name": f"{brand} {fake.word().capitalize()} {random.choice(['Pack','Bundle','Box','Can','Bottle'])}",
            "category": cat,
            "brand": brand,
            "package_size": random.choice(["100g", "200g", "500g", "1L", "250ml", "6-pack"]),
            "list_price": round(random.uniform(1.5, 35.0), 2),
            "launch_date": fake.date_between(start_date="-5y", end_date="-6m"),
            "is_active": random.choices([True, False], weights=[90, 10])[0],
        })
    return pd.DataFrame(rows)


# ── Store regions ─────────────────────────────────────────────────────────────

def generate_store_regions() -> pd.DataFrame:
    rows = []
    store_idx = 1
    for region, states in STATES_BY_REGION.items():
        for _ in range(NUM_STORES // len(STATES_BY_REGION)):
            state = random.choice(states)
            rows.append({
                "store_id": f"STORE-{store_idx:03d}",
                "store_name": f"{fake.city()} {random.choice(['Mart','Hub','Depot','Express'])}",
                "region": region,
                "state": state,
                "city": fake.city(),
                "demographic_segment": random.choice(DEMOGRAPHICS),
            })
            store_idx += 1
    return pd.DataFrame(rows)


# ── Sales transactions (with quality issues) ──────────────────────────────────

def _add_seasonality(date: datetime, category: str) -> float:
    """Seasonal demand multiplier."""
    month = date.month
    if category == "Beverages":
        return 1.0 + 0.4 * np.sin((month - 6) * np.pi / 6)   # peak summer
    if category == "Dairy":
        return 1.0 + 0.2 * np.cos((month - 1) * np.pi / 6)   # peak winter
    return 1.0 + 0.1 * np.sin(month * np.pi / 12)


def generate_transactions(
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    skus = catalog["sku"].tolist()
    store_ids = stores["store_id"].tolist()
    date_range = (END_DATE - START_DATE).days

    rows = []
    for i in range(NUM_TRANSACTIONS):
        txn_date = START_DATE + timedelta(days=random.randint(0, date_range))
        sku = random.choice(skus)
        cat = catalog.loc[catalog["sku"] == sku, "category"].values[0]
        list_price = catalog.loc[catalog["sku"] == sku, "list_price"].values[0]

        multiplier = _add_seasonality(txn_date, cat)
        quantity = max(1, int(np.random.poisson(lam=5 * multiplier)))
        # slight price variance (discounts / surcharges)
        unit_price = round(list_price * random.uniform(0.85, 1.10), 2)

        rows.append({
            "transaction_id": f"TXN-{i+1:06d}",
            "transaction_date": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
            "sku": sku,
            "quantity": quantity,
            "unit_price": unit_price,
            "store_id": random.choice(store_ids),
            "source_system": random.choice(SOURCE_SYSTEMS),
        })

    df = pd.DataFrame(rows)

    # ── Inject quality issues ─────────────────────────────────────────────────

    n = len(df)

    # 1. Null values (~3 % of rows missing quantity or price)
    null_idx = df.sample(frac=0.03).index
    df.loc[null_idx, "quantity"] = np.nan

    null_idx2 = df.sample(frac=0.02).index
    df.loc[null_idx2, "unit_price"] = np.nan

    # 2. Inconsistent date formats (~2 % rows use a different format)
    alt_fmt_idx = df.sample(frac=0.02).index
    df.loc[alt_fmt_idx, "transaction_date"] = pd.to_datetime(
        df.loc[alt_fmt_idx, "transaction_date"]
    ).dt.strftime("%d/%m/%Y")

    # 3. Duplicates (retry pattern — ~1 % rows duplicated)
    dup_rows = df.sample(frac=0.01)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. Negative quantities (data-entry errors)
    neg_idx = df.sample(frac=0.005).index
    df.loc[neg_idx, "quantity"] = -df.loc[neg_idx, "quantity"].abs()

    # 5. Inconsistent region casing in store_id column
    #    (simulates schema drift — some sources send STORE vs store)
    drift_idx = df.sample(frac=0.02).index
    df.loc[drift_idx, "store_id"] = df.loc[drift_idx, "store_id"].str.lower()

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(__file__)

    print("Generating product catalog...")
    catalog = generate_product_catalog()
    catalog.to_csv(os.path.join(out_dir, "product_catalog.csv"), index=False)
    print(f"  {len(catalog)} SKUs written.")

    print("Generating store regions...")
    stores = generate_store_regions()
    stores.to_csv(os.path.join(out_dir, "store_regions.csv"), index=False)
    print(f"  {len(stores)} stores written.")

    print("Generating transactions (with quality issues)...")
    txns = generate_transactions(catalog, stores)
    txns.to_csv(os.path.join(out_dir, "raw_transactions.csv"), index=False)
    print(f"  {len(txns)} rows written (including injected noise).")

    print("Done. Files saved to data/")


if __name__ == "__main__":
    main()
