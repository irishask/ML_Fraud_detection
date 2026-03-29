"""
eda.py — EDA functions for IEEE-CIS Fraud Detection.

Each public function corresponds to one logical EDA section.
Numerical calculations and their visualizations live in the same function.
All functions accept `train` (and optionally `test`) as DataFrames.
They print results and/or display matplotlib figures; return None unless
a computed value is needed downstream (documented in the docstring).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — Transaction-Level Analysis
# ════════════════════════════════════════════════════════════════════════════════

# ── 1.1 Target Imbalance ─────────────────────────────────────────────────────

def analyze_target_imbalance(train: pd.DataFrame) -> None:
    """
    Section 1.1 — Print class counts / percentages and show imbalance bar charts.
    Notebook cells: [08] + [09]
    """
    counts = train["isFraud"].value_counts()
    pcts   = train["isFraud"].value_counts(normalize=True)

    print("Target distribution (absolute):")
    print(counts)
    print("\nTarget distribution (%):")
    print((pcts * 100).round(1))

    fraud_rate = pcts[1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts.plot.bar(ax=axes[0], color=["#3b82f6", "#ef4444"])
    axes[0].set_title("Absolute Count")
    axes[0].set_xticklabels(["Legitimate (0)", "Fraud (1)"], rotation=0)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 5000, f"{v:,}", ha="center", fontweight="bold")

    pcts.plot.bar(ax=axes[1], color=["#3b82f6", "#ef4444"])
    axes[1].set_title("Percentage")
    axes[1].set_xticklabels(["Legitimate (0)", "Fraud (1)"], rotation=0)
    axes[1].set_ylabel("Fraction")
    for i, v in enumerate(pcts):
        axes[1].text(i, v + 0.005, f"{v:.2%}", ha="center", fontweight="bold")

    axes[0].axhline(y=counts[1], color="#ef4444", linestyle="--", alpha=0.5)
    axes[1].axhline(y=fraud_rate, color="#ef4444", linestyle="--", alpha=0.5,
                    label=f"Fraud rate = {fraud_rate:.2%}")
    axes[1].legend()

    plt.suptitle("Target Class Imbalance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 1.2 Time Patterns ────────────────────────────────────────────────────────

def analyze_time_range(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Show TransactionDT range in days for train / val / test splits.
    Called at the end of EDA after save_processed() creates the 3 splits.
    Replaces the old train-vs-Kaggle-test comparison (Kaggle test removed from project).
    """
    print("TransactionDT range (days):")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        lo = df["TransactionDT"].min() / 86400
        hi = df["TransactionDT"].max() / 86400
        print(f"  {name:5s}: day {lo:.0f} — day {hi:.0f}  ({len(df):,} rows)")

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"train": "#3b82f6", "val": "#10b981", "test": "#f59e0b"}
    for name, df, color in [("train", train, colors["train"]),
                             ("val",   val,   colors["val"]),
                             ("test",  test,  colors["test"])]:
        ax.hist(df["TransactionDT"] / 86400, bins=100, alpha=0.7,
                color=color, label=name.capitalize())

    ax.set_xlabel("Day")
    ax.set_ylabel("Transaction count")
    ax.set_title("Transaction Distribution Over Time — train / val / test (60/20/20)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def analyze_fraud_over_time(train: pd.DataFrame) -> None:
    """
    Section 1.2 (part 2) — Daily fraud rate and daily transaction volume over time.
    Notebook cell: [16]
    """
    daily_fraud = train.groupby("tx_day")["isFraud"].agg(["mean", "count"])
    daily_fraud.columns = ["fraud_rate", "tx_count"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(daily_fraud.index, daily_fraud["fraud_rate"], color="#ef4444", alpha=0.7)
    axes[0].axhline(y=train["isFraud"].mean(), color="black", linestyle="--", alpha=0.5,
                    label=f"Overall fraud rate = {train['isFraud'].mean():.4f}")
    axes[0].set_ylabel("Fraud Rate")
    axes[0].set_title("Daily Fraud Rate Over Time")
    axes[0].legend()

    axes[1].bar(daily_fraud.index, daily_fraud["tx_count"], color="#3b82f6", alpha=0.7)
    axes[1].set_ylabel("Transaction Count")
    axes[1].set_xlabel("Day")
    axes[1].set_title("Daily Transaction Volume")

    plt.tight_layout()
    plt.show()


def analyze_day_of_week(train: pd.DataFrame) -> None:
    """
    Section 1.2 (part 3) — Transaction volume and fraud rate by day of week.
    Notebook cell: [18]
    """
    dow_total      = train.groupby("tx_dow").size()
    dow_fraud      = train[train["isFraud"] == 1].groupby("tx_dow").size()
    dow_fraud_rate = train.groupby("tx_dow")["isFraud"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(dow_total.index, dow_total, color="#3b82f6", alpha=0.7, label="Legitimate")
    ax.bar(dow_fraud.index, dow_fraud, color="#ef4444", alpha=0.9, label="Fraud")
    for i, v in enumerate(dow_total):
        ax.text(i, v + 1000, f"{v:,}", ha="center", fontsize=8)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"Day {i}" for i in range(7)])
    ax.set_ylabel("Transaction Count")
    ax.set_title("Transaction Volume by Day of Week")
    ax.legend()

    ax = axes[1]
    ax.bar(dow_fraud_rate.index, dow_fraud_rate, color="#ef4444", alpha=0.8)
    ax.axhline(y=train["isFraud"].mean(), color="black", linestyle="--", alpha=0.5,
               label=f"Overall fraud rate = {train['isFraud'].mean():.4f}")
    for i, v in enumerate(dow_fraud_rate):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"Day {i}" for i in range(7)])
    ax.set_ylabel("Fraud Rate")
    ax.set_title("Fraud Rate by Day of Week")
    ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_hour_of_day(train: pd.DataFrame) -> None:
    """
    Section 1.2 (part 4) — Stacked 100% bar: fraud vs legitimate share by hour.
    Notebook cell: [20]
    """
    hour_fraud = train[train["isFraud"] == 1].groupby("tx_hour").size()
    hour_legit = train[train["isFraud"] == 0].groupby("tx_hour").size()
    hour_total = hour_fraud + hour_legit
    fraud_pct  = hour_fraud / hour_total * 100
    legit_pct  = hour_legit / hour_total * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(24), legit_pct, color="#b6e8b6", alpha=0.9, label="Legitimate")
    ax.bar(range(24), fraud_pct, bottom=legit_pct, color="#f4b4b4", alpha=0.9, label="Fraud")
    for i in range(24):
        ax.text(i, 101, f"{fraud_pct.iloc[i]:.1f}%", ha="center", fontsize=8,
                color="#cc0000", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("% of Transactions")
    ax.set_title("Fraud vs Legitimate Proportion by Hour (Stacked 100%)")
    ax.set_xticks(range(24))
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.show()


def analyze_day_of_month(train: pd.DataFrame) -> None:
    """
    Section 1.2 (part 5) — Transaction volume and fraud rate by day of month.
    Notebook cell: [22]
    """
    dom_total      = train.groupby("tx_dom").size()
    dom_fraud_rate = train.groupby("tx_dom")["isFraud"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(dom_total.index, dom_total, color="#b6e8b6", alpha=0.9)
    axes[0].set_xlabel("Day of Month")
    axes[0].set_ylabel("Transaction Count")
    axes[0].set_title("Transaction Volume by Day of Month")

    axes[1].bar(dom_fraud_rate.index, dom_fraud_rate, color="#f4b4b4", alpha=0.9)
    axes[1].axhline(y=train["isFraud"].mean(), color="black", linestyle="--", alpha=0.5,
                    label=f"Overall fraud rate = {train['isFraud'].mean():.4f}")
    for i, v in enumerate(dom_fraud_rate):
        axes[1].text(dom_fraud_rate.index[i], v + 0.001, f"{v:.3f}", ha="center",
                     fontsize=7, color="#cc0000", fontweight="bold")
    axes[1].set_xlabel("Day of Month")
    axes[1].set_ylabel("Fraud Rate")
    axes[1].set_title("Fraud Rate by Day of Month")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ── 1.3 Transaction Amount ───────────────────────────────────────────────────

def analyze_transaction_amount(train: pd.DataFrame) -> None:
    """
    Section 1.3 — Print fraud vs legit amount statistics; show full and log
    distributions.
    Notebook cells: [25] + [26]
    """
    print("TransactionAmt — Fraud vs Legitimate:")
    for label, group in train.groupby("isFraud")["TransactionAmt"]:
        name = "Fraud" if label == 1 else "Legitimate"
        print(f"\n  {name}:")
        print(f"    Count:  {group.count():>10,}")
        print(f"    Mean:   ${group.mean():>10,.2f}")
        print(f"    Median: ${group.median():>10,.2f}")
        print(f"    Min:    ${group.min():>10,.2f}")
        print(f"    Max:    ${group.max():>10,.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.hist(train[train["isFraud"] == 0]["TransactionAmt"], bins=100, alpha=0.7,
            color="#b3cde3", label="Legitimate", density=True)
    ax.hist(train[train["isFraud"] == 1]["TransactionAmt"], bins=100, alpha=0.7,
            color="#f4b4b4", label="Fraud", density=True)
    ax.set_xlabel("TransactionAmt ($)")
    ax.set_ylabel("Density")
    ax.set_title("Full Distribution (Capped at $1,000)")
    ax.set_xlim(0, 1000)
    ax.legend()

    ax = axes[1]
    ax.hist(np.log1p(train[train["isFraud"] == 0]["TransactionAmt"]), bins=100, alpha=0.7,
            color="#b3cde3", label="Legitimate", density=True)
    ax.hist(np.log1p(train[train["isFraud"] == 1]["TransactionAmt"]), bins=100, alpha=0.7,
            color="#f4b4b4", label="Fraud", density=True)
    ax.set_xlabel("log(TransactionAmt + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Log-Transformed Distribution")
    ax.legend()

    plt.suptitle("TransactionAmt: Fraud vs Legitimate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 1.4 ProductCD ────────────────────────────────────────────────────────────

def analyze_product_cd(train: pd.DataFrame) -> None:
    """
    Section 1.4 — Print ProductCD breakdown; show stacked absolute count bar chart.
    Notebook cells: [29] + [30]
    """
    prod_total      = train.groupby("ProductCD").size()
    prod_fraud_rate = train.groupby("ProductCD")["isFraud"].mean()

    print("ProductCD breakdown:")
    for prod in prod_total.index:
        count = prod_total[prod]
        fraud = prod_fraud_rate[prod]
        pct   = count / len(train) * 100
        print(f"  {prod} — Transactions: {count:>8,} ({pct:5.1f}%)   Fraud rate: {fraud:.2%}")

    prod_fraud = train[train["isFraud"] == 1].groupby("ProductCD").size()
    prod_legit = train[train["isFraud"] == 0].groupby("ProductCD").size()
    fraud_rate = prod_fraud / (prod_fraud + prod_legit) * 100
    products   = prod_fraud.index

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(products, prod_legit, color="#b6e8b6", alpha=0.9, label="Legitimate")
    ax.bar(products, prod_fraud, bottom=prod_legit, color="#f4b4b4", alpha=0.9, label="Fraud")
    for i, prod in enumerate(products):
        total = prod_legit[prod] + prod_fraud[prod]
        ax.text(i, total + 5000, f"Fraud: {fraud_rate[prod]:.1f}%", ha="center",
                fontsize=10, color="#cc0000", fontweight="bold")
    ax.set_xlabel("ProductCD")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Fraud vs Legitimate by ProductCD (Absolute Count)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 1.5 Device Type ──────────────────────────────────────────────────────────

def analyze_device_type(train: pd.DataFrame) -> None:
    """
    Section 1.5 (part 1) — Print device type fraud stats; show pie charts
    (share of all tx vs share of fraud tx).
    Notebook cells: [33] + [34]
    """
    device_stats = train.groupby("DeviceType")["isFraud"].agg(["count", "mean"])
    print("DeviceType Analysis:")
    for device in device_stats.index:
        count = device_stats.loc[device, "count"]
        fraud = device_stats.loc[device, "mean"]
        print(f"  {device:10s} — Transactions: {count:>8,}   Fraud rate: {fraud:.4f} ({fraud:.2%})")
    missing = train["DeviceType"].isna().sum()
    print(f"\n  No device info — Transactions: {missing:>8,}   "
          f"({train['DeviceType'].isna().mean():.1%} of all data)")

    fraud_desktop = train[train["DeviceType"] == "desktop"]["isFraud"].sum()
    fraud_mobile  = train[train["DeviceType"] == "mobile"]["isFraud"].sum()
    fraud_nodev   = train[train["DeviceType"].isna()]["isFraud"].sum()

    labels = ["Desktop", "Mobile", "No device info"]
    colors = ["#c4b7d4", "#d3d3d3", "#b3cde3"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tx_counts = [85165, 55645, 449730]
    wedges, texts, auto = axes[0].pie(
        tx_counts, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(tx_counts) / 100):,})",
        explode=(0.02, 0.02, 0.05), startangle=90, textprops={"fontsize": 10}
    )
    for t in auto:
        t.set_fontweight("bold"); t.set_color("#333333")
    axes[0].set_title("Share of All Transactions", fontsize=13, fontweight="bold")

    fraud_counts = [fraud_desktop, fraud_mobile, fraud_nodev]
    wedges, texts, auto = axes[1].pie(
        fraud_counts, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(fraud_counts) / 100):,})",
        explode=(0.02, 0.02, 0.05), startangle=90, textprops={"fontsize": 10}
    )
    for t in auto:
        t.set_fontweight("bold"); t.set_color("#333333")
    axes[1].set_title("Share of Fraud Transactions", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()


def analyze_device_product_cross(train: pd.DataFrame) -> None:
    """
    Section 1.5 (part 2) — ProductCD × DeviceType crosstab + fraud rate bar chart.
    Notebook cells: [36] + [37] + [38]
    """
    fraud_cross = (train.groupby(["ProductCD", "DeviceType_filled"])["isFraud"]
                   .agg(["mean", "count"])
                   .reset_index())
    fraud_cross.columns = ["ProductCD", "DeviceType_filled", "fraud_rate", "tx_count"]
    fraud_cross = fraud_cross[fraud_cross["tx_count"] > 100]

    cross = pd.crosstab(train["ProductCD"], train["DeviceType_filled"], normalize="index") * 100
    print("DeviceType distribution within each ProductCD (%):")
    print(cross.round(1))

    prod_order    = (train.groupby("ProductCD")["isFraud"].mean()
                     .sort_values(ascending=False).index)
    pivot         = fraud_cross.pivot(index="ProductCD", columns="DeviceType_filled",
                                      values="fraud_rate") * 100
    pivot         = pivot.reindex(prod_order)
    overall_fraud = train.groupby("ProductCD")["isFraud"].mean().reindex(prod_order) * 100
    colors        = {"desktop": "#c4b7d4", "mobile": "#d3d3d3", "No device info": "#b3cde3"}

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.bar(ax=ax, color=[colors[c] for c in pivot.columns], alpha=0.9, width=0.7)
    for i, prod in enumerate(prod_order):
        ax.plot(i, overall_fraud[prod], marker="D", color="#cc0000", markersize=10, zorder=5)
        ax.text(i + 0.35, overall_fraud[prod],
                f"Fraud Rate: {overall_fraud[prod]:.1f}%",
                fontsize=10, fontweight="bold", color="#cc0000", va="center")
    ax.set_xlabel("ProductCD")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by ProductCD × DeviceType (Sorted by Fraud Rate)")
    ax.set_xticklabels(prod_order, rotation=0)
    ax.axhline(y=3.5, color="black", linestyle="--", alpha=0.3, label="Dataset avg (3.5%)")
    ax.legend(title="DeviceType")
    plt.tight_layout()
    plt.show()


# ── 1.6 Email Domains ────────────────────────────────────────────────────────

def analyze_email_domains(train: pd.DataFrame) -> None:
    """
    Section 1.6 — NaN%/unique for P and R emaildomain; top-10 fraud rates + charts.
    Notebook cells: [41] + [42] + [43]
    """
    for col in ["P_emaildomain", "R_emaildomain"]:
        nunique = train[col].nunique()
        nan_pct = train[col].isna().mean() * 100
        print(f"  {col}: unique: {nunique:>4}   NaN: {nan_pct:.1f}%")

    for col in ["P_emaildomain", "R_emaildomain"]:
        print(f"\n{col} — Top 10 by transaction count:")
        stats = (train.groupby(col)["isFraud"]
                 .agg(["count", "mean"])
                 .sort_values("count", ascending=False)
                 .head(10))
        for domain, row in stats.iterrows():
            pct = row["count"] / len(train) * 100
            print(f"  {domain:25s} — Tx: {row['count']:>8,.0f} ({pct:5.1f}%)"
                  f"   Fraud rate: {row['mean']:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    chart_colors = {"P_emaildomain": "#b3cde3", "R_emaildomain": "#c4b7d4"}

    for ax, col in zip(axes, ["P_emaildomain", "R_emaildomain"]):
        stats = (train.groupby(col)["isFraud"]
                 .agg(["count", "mean"])
                 .sort_values("count", ascending=False)
                 .head(10)
                 .sort_values("mean", ascending=True))
        fraud_pct = stats["mean"] * 100
        ax.barh(range(len(fraud_pct)), fraud_pct, color=chart_colors[col], alpha=0.9)
        for i, (domain, row) in enumerate(stats.iterrows()):
            ax.text(fraud_pct[domain] + 0.2, i,
                    f"{fraud_pct[domain]:.1f}%  (n={row['count']:,.0f})",
                    va="center", fontsize=9, color="#cc0000", fontweight="bold")
        ax.axvline(x=3.5, color="black", linestyle="--", alpha=0.4, label="Dataset avg (3.5%)")
        ax.set_yticks(range(len(fraud_pct)))
        ax.set_yticklabels(stats.index)
        ax.set_xlabel("Fraud Rate (%)")
        label = "Purchaser" if col == "P_emaildomain" else "Recipient"
        ax.set_title(f"Fraud Rate by {label} Email Domain (Top 10)")
        ax.legend()

    plt.tight_layout()
    plt.show()


# ── 1.7 Card Attributes ──────────────────────────────────────────────────────

def analyze_card_attributes(train: pd.DataFrame) -> None:
    """
    Section 1.7 — card1-6 dtype/NaN/unique; card4/card6 breakdown + bar charts.
    Notebook cells: [46] + [47] + [49] + [50]
    """
    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
    for col in card_cols:
        print(f"  {col}: {str(train[col].dtype):10s}  unique: {train[col].nunique():>6,}"
              f"   NaN: {train[col].isna().mean() * 100:.1f}%")

    print()
    display(train[["TransactionID", "card1", "card2", "card3",
                   "card4", "card5", "card6", "isFraud"]].head(10))

    for col in ["card4", "card6"]:
        print(f"\n{col} breakdown:")
        stats = train.groupby(col)["isFraud"].agg(["count", "mean"])
        for val in stats.index:
            count = stats.loc[val, "count"]
            fraud = stats.loc[val, "mean"]
            pct   = count / len(train) * 100
            print(f"  {val:15s} — Tx: {count:>8,} ({pct:5.1f}%)   Fraud rate: {fraud:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["card4", "card6"]):
        stats = (train.groupby(col)["isFraud"]
                 .agg(["count", "mean"])
                 .sort_values("mean", ascending=False))
        stats     = stats[stats["count"] > 100]
        fraud_pct = stats["mean"] * 100
        ax.bar(range(len(fraud_pct)), fraud_pct, color="#f4b4b4", alpha=0.9)
        for i, (val, pct) in enumerate(fraud_pct.items()):
            ax.text(i, pct + 0.2, f"{pct:.1f}%", ha="center", fontsize=11,
                    color="#cc0000", fontweight="bold")
            ax.text(i, pct / 2, f"n={stats.loc[val, 'count']:,.0f}", ha="center",
                    fontsize=9, color="#333333")
        ax.axhline(y=3.5, color="black", linestyle="--", alpha=0.4, label="Dataset avg (3.5%)")
        ax.set_xticks(range(len(fraud_pct)))
        ax.set_xticklabels(fraud_pct.index, rotation=0)
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title(f"Fraud Rate by {col}")
        ax.legend()

    plt.tight_layout()
    plt.show()


# ── 1.8 Missing Values ───────────────────────────────────────────────────────

def analyze_missing_values(train: pd.DataFrame) -> None:
    """
    Section 1.8 — NaN summary by level; bar chart; detailed column breakdown.
    Notebook cells: [53] + [54] + [55]
    """
    nan_pct = (train.isnull().mean() * 100).sort_values(ascending=False)

    print("Missing Values Summary:")
    print(f"  Columns with 0% NaN:     {(nan_pct == 0).sum()}")
    print(f"  Columns with 0-25% NaN:  {((nan_pct > 0)  & (nan_pct <= 25)).sum()}")
    print(f"  Columns with 25-50% NaN: {((nan_pct > 25) & (nan_pct <= 50)).sum()}")
    print(f"  Columns with 50-75% NaN: {((nan_pct > 50) & (nan_pct <= 75)).sum()}")
    print(f"  Columns with 75%+ NaN:   {(nan_pct > 75).sum()}")
    print(f"\n  Total columns: {len(nan_pct)}")

    nan_groups = {
        "0% NaN":  (nan_pct == 0).sum(),
        "0–25%":   ((nan_pct > 0)  & (nan_pct <= 25)).sum(),
        "25–50%":  ((nan_pct > 25) & (nan_pct <= 50)).sum(),
        "50–75%":  ((nan_pct > 50) & (nan_pct <= 75)).sum(),
        "75%+":    (nan_pct > 75).sum(),
    }
    colors = ["#b6e8b6", "#b3cde3", "#ffe0b2", "#f4b4b4", "#c4b7d4"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(nan_groups.keys(), nan_groups.values(), color=colors, alpha=0.9)
    for i, (label, val) in enumerate(nan_groups.items()):
        ax.text(i, val + 3, f"{val}", ha="center", fontsize=12,
                color="#333333", fontweight="bold")
    ax.set_xlabel("NaN Level")
    ax.set_ylabel("Number of Columns")
    ax.set_title("Missing Values Distribution Across 439 Columns")
    plt.tight_layout()
    plt.show()

    def _column_details(df, nan_range_label, nan_min, nan_max):
        cols = (nan_pct[nan_pct == 0].index.tolist() if nan_min == -1
                else nan_pct[(nan_pct > nan_min) & (nan_pct <= nan_max)].index.tolist())
        if not cols:
            return
        v_cols     = [c for c in cols if c.startswith("V")]
        id_cols    = [c for c in cols if c.startswith("id_")]
        other_cols = [c for c in cols if c not in v_cols and c not in id_cols]
        print(f"\n{'='*80}")
        print(f"  {nan_range_label}: {len(cols)} columns")
        print(f"{'='*80}")
        if v_cols:
            print(f"\n  V-columns ({len(v_cols)}): "
                  f"V{min(int(c[1:]) for c in v_cols)}–V{max(int(c[1:]) for c in v_cols)}")
        if id_cols:
            print(f"  Identity columns ({len(id_cols)}): {', '.join(sorted(id_cols))}")
        if other_cols:
            print(f"\n  Other columns ({len(other_cols)}):")
            rows = []
            for col in other_cols:
                top_vals = df[col].dropna().value_counts().head(5).index.tolist()
                rows.append({
                    "column":     col,
                    "NaN %":      f"{nan_pct[col]:.1f}%",
                    "dtype":      str(df[col].dtype),
                    "unique":     df[col].nunique(),
                    "top 5 vals": str(top_vals)[:60],
                })
            display(pd.DataFrame(rows).set_index("column"))

    _column_details(train, "0% NaN",    -1,   0)
    _column_details(train, "0–25% NaN",  0,  25)
    _column_details(train, "25–50% NaN", 25, 50)
    _column_details(train, "50–75% NaN", 50, 75)
    _column_details(train, "75%+ NaN",   75, 100)


# ── 1.9 Correlations with isFraud ────────────────────────────────────────────

def analyze_correlations(train: pd.DataFrame) -> pd.Series:
    """
    Section 1.9 — Print top-20 and bottom-5 |r| with isFraud; horizontal bar chart.
    Returns the full correlations Series for downstream use (e.g. D-column analysis).
    Notebook cells: [58] + [59]
    """
    numeric_cols = train.select_dtypes(include="number").columns.tolist()
    numeric_cols.remove("isFraud")
    correlations = train[numeric_cols].corrwith(train["isFraud"]).abs().sort_values(ascending=False)

    print("Top 20 features most correlated with isFraud (absolute value):")
    for col, corr in correlations.head(20).items():
        print(f"  {col:25s}  |r| = {corr:.4f}")
    print("\nBottom 5 (least correlated):")
    for col, corr in correlations.tail(5).items():
        print(f"  {col:25s}  |r| = {corr:.4f}")

    top20  = correlations.head(20).sort_values(ascending=True)
    colors = ["#c4b7d4" if col.startswith("V") else "#b3cde3" for col in top20.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20.values, color=colors, alpha=0.9)
    for i, (col, val) in enumerate(top20.items()):
        ax.text(val + 0.003, i, f"{val:.4f}", va="center", fontsize=9, color="#333333")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index)
    ax.set_xlabel("|Correlation| with isFraud")
    ax.set_title("Top 20 Features Most Correlated with isFraud")
    plt.tight_layout()
    plt.show()

    return correlations


# ── 1.10 D-Columns ───────────────────────────────────────────────────────────

def analyze_d_columns_basic(train: pd.DataFrame) -> list:
    """
    Section 1.10.1 — dtype / NaN% / neg% / min / max / mean / median for D1-D15.
    Returns d_cols_present (list) for use in all subsequent D-column functions.
    Notebook cell: [63]
    """
    d_cols = [f"D{i}" for i in range(1, 16)]
    print(f"{'Col':<6} {'dtype':>10}  {'NaN %':>7}  {'neg %':>7}  "
          f"{'min':>10}  {'max':>10}  {'mean':>10}  {'median':>10}")
    print("-" * 85)
    for col in d_cols:
        if col not in train.columns:
            print(f"  {col}: NOT IN DATASET")
            continue
        s     = train[col].dropna()
        nan_p = train[col].isna().mean() * 100
        neg_p = (s < 0).mean() * 100
        print(f"  {col:<4}  {str(train[col].dtype):>10}  {nan_p:>6.1f}%  {neg_p:>6.1f}%  "
              f"{s.min():>10.1f}  {s.max():>10.1f}  {s.mean():>10.1f}  {s.median():>10.1f}")
    return [c for c in d_cols if c in train.columns]


def analyze_d_columns_nan(train: pd.DataFrame, d_cols_present: list) -> None:
    """
    Section 1.10.2 — Bar chart of D-column NaN% with 25/50/75% threshold lines.
    Notebook cell: [66]
    """
    nan_pcts = [train[c].isna().mean() * 100 for c in d_cols_present]

    fig, ax = plt.subplots(figsize=(12, 4))
    bar_colors = ["#f4b4b4" if v > 75 else "#ffe0b2" if v > 25 else "#b3cde3"
                  for v in nan_pcts]
    bars = ax.bar(d_cols_present, nan_pcts, color=bar_colors, alpha=0.9)
    for bar, val in zip(bars, nan_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=9, color="#333333")
    ax.axhline(y=75, color="red",    linestyle="--", alpha=0.5, label="75% — drop threshold")
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.4, label="50% threshold")
    ax.axhline(y=25, color="green",  linestyle="--", alpha=0.4, label="25% threshold")
    ax.set_ylabel("Missing Values (%)")
    ax.set_title("D-Columns: Missing Values (red bars = >75% NaN → candidates for drop)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    drop_candidates = [c for c, v in zip(d_cols_present, nan_pcts) if v > 75]
    keep_candidates = [c for c, v in zip(d_cols_present, nan_pcts) if v <= 75]
    print(f"Drop candidates (>75% NaN): {drop_candidates}")
    print(f"Keep candidates (≤75% NaN): {keep_candidates}")


def analyze_d_columns_correlations(train: pd.DataFrame, d_cols_present: list) -> None:
    """
    Section 1.10.3 — Correlation heatmap for D-columns + isFraud; sorted |r| list.
    Notebook cell: [69]
    """
    d_corr_cols = d_cols_present + ["isFraud"]
    corr_matrix = train[d_corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#cccccc", ax=ax)
    ax.set_title("D-Columns Correlation Matrix (including isFraud)")
    plt.tight_layout()
    plt.show()

    print("\nD-column correlations with isFraud (sorted by |r|):")
    d_fraud_corr = corr_matrix["isFraud"].drop("isFraud").abs().sort_values(ascending=False)
    for col, val in d_fraud_corr.items():
        nan_p = train[col].isna().mean() * 100
        print(f"  {col}: |r| = {val:.4f}   NaN: {nan_p:.0f}%")


def analyze_d_columns_distributions(train: pd.DataFrame, d_cols_present: list) -> None:
    """
    Section 1.10.4 — Raw D-column distributions fraud vs legit (p1-p99 for display).
    Notebook cell: [72]
    """
    ncols_plot = 3
    nrows_plot = (len(d_cols_present) + ncols_plot - 1) // ncols_plot
    fig, axes  = plt.subplots(nrows_plot, ncols_plot, figsize=(15, nrows_plot * 3.5))
    axes       = axes.flatten()

    for idx, col in enumerate(d_cols_present):
        ax         = axes[idx]
        legit      = train[train["isFraud"] == 0][col].dropna()
        fraud      = train[train["isFraud"] == 1][col].dropna()
        p1, p99    = legit.quantile(0.01), legit.quantile(0.99)
        legit_plot = legit.clip(lower=p1, upper=p99)
        fraud_plot = fraud.clip(lower=p1, upper=p99)
        ax.hist(legit_plot, bins=60, alpha=0.6, color="#b3cde3", density=True, label="Legitimate")
        ax.hist(fraud_plot, bins=60, alpha=0.6, color="#f4b4b4", density=True, label="Fraud")
        nan_p = train[col].isna().mean() * 100
        neg_p = (train[col].dropna() < 0).mean() * 100
        title = f"{col}  (NaN:{nan_p:.0f}%"
        if neg_p > 0:
            title += f"  neg:{neg_p:.1f}%"
        title += ")"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Raw value (p1–p99 display range)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(len(d_cols_present), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("D-Columns: Raw Distribution — Fraud vs Legitimate\n"
                 "(clipped to p1–p99 for display; negatives shown where present)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def analyze_d_columns_quantile_fraud(train: pd.DataFrame, d_cols_present: list,
                                     n_bins: int = 10) -> None:
    """
    Section 1.10.5 — Fraud rate by quantile bin for each D-column.
    n_bins: number of quantile bins (default 10).
    Notebook cell: [75]
    """
    ncols_plot = 3
    nrows_plot = (len(d_cols_present) + ncols_plot - 1) // ncols_plot
    fig, axes  = plt.subplots(nrows_plot, ncols_plot, figsize=(15, nrows_plot * 3.5))
    axes       = axes.flatten()

    for idx, col in enumerate(d_cols_present):
        ax  = axes[idx]
        tmp = train[[col, "isFraud"]].dropna(subset=[col]).copy()
        try:
            tmp["bin"]  = pd.qcut(tmp[col], q=n_bins, duplicates="drop")
            fraud_rate  = tmp.groupby("bin", observed=True)["isFraud"].mean() * 100
            colors_bars = ["#f4b4b4" if v > 5 else "#b3cde3" for v in fraud_rate.values]
            ax.bar(range(len(fraud_rate)), fraud_rate.values, color=colors_bars, alpha=0.9)
            for i, v in enumerate(fraud_rate.values):
                if v > 5:
                    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=7,
                            color="#cc0000", fontweight="bold")
            ax.axhline(y=3.5, color="black", linestyle="--", alpha=0.4, label="Avg 3.5%")
            ax.set_xticks([])
            ax.set_ylabel("Fraud Rate (%)")
            nan_p = train[col].isna().mean() * 100
            ax.set_title(f"{col}  (NaN:{nan_p:.0f}%  bins:{len(fraud_rate)})", fontsize=10)
            ax.legend(fontsize=8)
        except Exception as e:
            ax.set_title(f"{col} — skipped: {e}", fontsize=9)

    for idx in range(len(d_cols_present), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"D-Columns: Fraud Rate by Quantile ({n_bins} bins)\n"
                 "Red bars = fraud rate >5% (above average)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — Customer-Level Analysis  (UID = card1 + addr1)
# ════════════════════════════════════════════════════════════════════════════════

# ── 2.1 UID Coverage ─────────────────────────────────────────────────────────

def analyze_uid_coverage(train: pd.DataFrame) -> None:
    """
    Section 2.1 — Coverage and cardinality of card1+addr1 as grouping key.
    Notebook cell: [79]
    """
    n_total     = len(train)
    n_both      = (train["card1"].notna() & train["addr1"].notna()).sum()
    n_card1_nan = train["card1"].isna().sum()
    n_addr1_nan = train["addr1"].isna().sum()

    print("UID = card1 + addr1 — Coverage:")
    print(f"  Total transactions       : {n_total:>10,}")
    print(f"  card1 NaN                : {n_card1_nan:>10,}  ({n_card1_nan/n_total:.2%})")
    print(f"  addr1 NaN                : {n_addr1_nan:>10,}  ({n_addr1_nan/n_total:.2%})")
    print(f"  Both non-NaN (full UID)  : {n_both:>10,}  ({n_both/n_total:.2%})")
    print(f"  UID incomplete           : {n_total-n_both:>10,}  ({(n_total-n_both)/n_total:.2%})")

    uid_c1    = train["card1"].nunique()
    uid_c1_a1 = train.groupby(["card1", "addr1"], observed=True).ngroups
    print("\nCardinality:")
    print(f"  card1 alone              : {uid_c1:>10,} groups  (avg {n_total/uid_c1:.0f} tx/group)")
    print(f"  card1 + addr1            : {uid_c1_a1:>10,} groups  (avg {n_total/uid_c1_a1:.0f} tx/group)")
    print(f"  Increase                 : {uid_c1_a1/uid_c1:.2f}x more groups with addr1")


# ── 2.2 Group Size Distribution ──────────────────────────────────────────────

def analyze_group_size_distribution(train: pd.DataFrame) -> None:
    """
    Section 2.2 — Group size stats + histogram + cumulative coverage chart.
    Notebook cell: [82]
    """
    group_sizes = (train.dropna(subset=["card1", "addr1"])
                   .groupby(["card1", "addr1"])
                   .size()
                   .sort_values())

    print("Group size distribution — card1+addr1:")
    print(f"  Total groups             : {len(group_sizes):>10,}")
    for lo, hi, lbl in [(1, 1, "= 1 tx (no history)"), (2, 5, "2–5 tx"),
                         (6, 20, "6–20 tx"), (21, 10**9, ">20 tx")]:
        n = ((group_sizes >= lo) & (group_sizes <= hi)).sum()
        print(f"  Groups {lbl:<22}: {n:>10,}  ({n/len(group_sizes):.1%})")
    print(f"  Median group size        : {group_sizes.median():>10.0f}")
    print(f"  Mean group size          : {group_sizes.mean():>10.1f}")
    print(f"  Max group size           : {group_sizes.max():>10,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(group_sizes.clip(upper=50), bins=50, color="#b3cde3", alpha=0.9)
    ax.axvline(x=1, color="red",    linestyle="--", alpha=0.7, label="1 tx (no history)")
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7, label="5 tx (min useful)")
    ax.set_xlabel("Transactions per group (capped at 50)")
    ax.set_ylabel("Number of groups")
    ax.set_title("Group Size Distribution — card1+addr1")
    ax.legend()

    ax = axes[1]
    cum = group_sizes.cumsum() / group_sizes.sum() * 100
    ax.plot(range(len(cum)), cum.values, color="#1F4E79", linewidth=2)
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.6, label="50% of transactions")
    ax.axhline(y=80, color="red",    linestyle="--", alpha=0.6, label="80% of transactions")
    ax.set_xlabel("Groups (sorted by size, smallest first)")
    ax.set_ylabel("Cumulative % of transactions")
    ax.set_title("Cumulative Transaction Coverage by Group Size")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ── 2.3 Fraud Concentration ──────────────────────────────────────────────────

def analyze_fraud_concentration(train: pd.DataFrame) -> None:
    """
    Section 2.3 — How well does card1+addr1 isolate fraud? Card compromise pattern.
    Notebook cell: [85]
    """
    fraud_by_group = (train.dropna(subset=["card1", "addr1"])
                      .groupby(["card1", "addr1"])["isFraud"]
                      .agg(["sum", "mean", "count"])
                      .rename(columns={"sum": "fraud_count",
                                       "mean": "fraud_rate",
                                       "count": "tx_count"}))

    n_gr      = len(fraud_by_group)
    has_fraud = (fraud_by_group["fraud_count"] > 0).sum()
    no_fraud  = (fraud_by_group["fraud_count"] == 0).sum()
    all_fraud = (fraud_by_group["fraud_rate"] == 1.0).sum()

    print("Fraud concentration — card1+addr1:")
    print(f"  Total groups             : {n_gr:>10,}")
    print(f"  Groups with ANY fraud    : {has_fraud:>10,}  ({has_fraud/n_gr:.2%})")
    print(f"  Groups with ZERO fraud   : {no_fraud:>10,}  ({no_fraud/n_gr:.2%})")
    print(f"  Groups 100% fraud        : {all_fraud:>10,}  ({all_fraud/n_gr:.2%})")

    fraud_grps   = fraud_by_group[fraud_by_group["fraud_count"] > 0]
    isolated     = (fraud_grps["fraud_count"] == 1).sum()
    clustered    = (fraud_grps["fraud_count"] >  1).sum()
    clustered_tx = fraud_grps[fraud_grps["fraud_count"] > 1]["fraud_count"].sum()
    print()
    print(f"  Fraud groups — 1 tx      : {isolated:>10,}  ({isolated/has_fraud:.1%})  ← isolated incident")
    print(f"  Fraud groups — 2+ tx     : {clustered:>10,}  ({clustered/has_fraud:.1%})  ← card compromise")
    print(f"  % of all fraud in 2+ grps: {clustered_tx/train['isFraud'].sum():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(fraud_grps["fraud_rate"], bins=30, color="#f4b4b4", alpha=0.9)
    ax.axvline(x=1.0, color="red",    linestyle="--", alpha=0.7, label="100% fraud group")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="50% fraud rate")
    ax.set_xlabel("Fraud rate within group")
    ax.set_ylabel("Number of groups")
    ax.set_title("Fraud Rate Distribution\n(groups with ≥1 fraud tx)")
    ax.legend()

    ax = axes[1]
    ax.hist(fraud_grps["fraud_count"].clip(upper=15), bins=15, color="#f4b4b4", alpha=0.9)
    ax.set_xlabel("Fraud transactions per group (capped at 15)")
    ax.set_ylabel("Number of groups")
    ax.set_title("Fraud Count per Group\n(card compromise pattern)")

    plt.tight_layout()
    plt.show()


# ── 2.4 card1 Alone vs card1+addr1 ───────────────────────────────────────────

def analyze_uid_addr1_value(train: pd.DataFrame) -> None:
    """
    Section 2.4 — Compare card1 alone vs card1+addr1: variance and purity metrics.
    Notebook cell: [88]
    """
    results = {}
    for label, cols in [("card1", ["card1"]), ("card1+addr1", ["card1", "addr1"])]:
        df = train.dropna(subset=cols).groupby(cols)["isFraud"].agg(["mean", "count"])
        df.columns = ["fraud_rate", "tx_count"]
        results[label] = df

    print(f"{'Metric':<38} {'card1':>14} {'card1+addr1':>14}")
    print("-" * 68)
    for metric, fn in [
        ("Unique groups",           lambda d: f"{len(d):,}"),
        ("Avg tx/group",            lambda d: f"{d['tx_count'].mean():.1f}"),
        ("fraud_rate variance",     lambda d: f"{d['fraud_rate'].var():.6f}"),
        ("% groups fraud_rate = 0", lambda d: f"{(d['fraud_rate']==0).mean():.1%}"),
        ("% groups fraud_rate = 1", lambda d: f"{(d['fraud_rate']==1.0).mean():.1%}"),
        ("% groups mixed fraud",    lambda d: f"{((d['fraud_rate']>0)&(d['fraud_rate']<1)).mean():.1%}"),
    ]:
        print(f"  {metric:<36} {fn(results['card1']):>14} {fn(results['card1+addr1']):>14}")

    colors = {"card1": "#b3cde3", "card1+addr1": "#c4b7d4"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (label, df) in zip(axes, results.items()):
        ax.hist(df["fraud_rate"], bins=40, color=colors[label], alpha=0.9)
        ax.set_xlabel("Fraud rate per group")
        ax.set_ylabel("Number of groups")
        ax.set_title(f"{label}  ({len(df):,} groups)\n"
                     f"fraud_rate variance = {df['fraud_rate'].var():.5f}")
        ax.axvline(x=0.035, color="black", linestyle="--", alpha=0.4, label="avg 3.5%")
        ax.legend()
    plt.tight_layout()
    plt.show()


# ── 2.5 D1 Stability ─────────────────────────────────────────────────────────

def analyze_d1_stability(train: pd.DataFrame) -> pd.DataFrame:
    """
    Section 2.5 — Test A (raw D1 std per group) + Test B (anchor drift first vs last).
    Cleans up D1_anchor column before and after.
    Returns train with D1_anchor dropped (clean state).
    Notebook cell: [91]
    """
    if "D1_anchor" in train.columns:
        train = train.drop(columns=["D1_anchor"])

    d1_std = (train.dropna(subset=["D1", "card1", "addr1"])
              .groupby(["card1", "addr1"])["D1"]
              .std().dropna())

    print("TEST A — Std of raw D1 per group:")
    print(f"  Groups (≥2 tx)           : {len(d1_std):,}")
    print(f"  Median std               : {d1_std.median():.2f} days")
    print(f"  % std = 0 (identical)    : {(d1_std==0).mean()*100:.1f}%")
    print(f"  % std < 1 day            : {(d1_std<1).mean()*100:.1f}%")
    print(f"  % std < 7 days           : {(d1_std<7).mean()*100:.1f}%")
    print(f"  % std > 30 days          : {(d1_std>30).mean()*100:.1f}%")
    print()

    train["D1_anchor"] = (train["TransactionDT"] / 86400) - train["D1"]
    anchor_fl = (train.dropna(subset=["D1", "card1", "addr1"])
                 .sort_values("TransactionDT")
                 .groupby(["card1", "addr1"])["D1_anchor"]
                 .agg(["first", "last", "count"])
                 .query("count >= 2"))
    anchor_fl["drift"] = (anchor_fl["last"] - anchor_fl["first"]).abs()

    print("TEST B — Anchor drift |first_anchor - last_anchor| per group:")
    print(f"  Groups (≥2 tx)           : {len(anchor_fl):,}")
    print(f"  Median drift             : {anchor_fl['drift'].median():.2f} days")
    print(f"  % drift = 0              : {(anchor_fl['drift']==0).mean()*100:.1f}%  ← perfectly constant")
    print(f"  % drift < 1 day          : {(anchor_fl['drift']<1).mean()*100:.1f}%  ← near-constant")
    print(f"  % drift < 7 days         : {(anchor_fl['drift']<7).mean()*100:.1f}%")
    print(f"  % drift > 30 days        : {(anchor_fl['drift']>30).mean()*100:.1f}%")
    print()
    pct = (anchor_fl["drift"] < 1).mean()
    if   pct > 0.80: print("  ✅ D1 IS stable → safe to use in UID: card1+addr1+D1")
    elif pct > 0.50: print("  ⚠️  D1 PARTIALLY stable → usable for stable subset only")
    else:            print("  ❌ D1 NOT stable → dynamic per transaction, do NOT use in UID key")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(d1_std.clip(upper=100), bins=80, color="#b3cde3", alpha=0.9)
    ax.axvline(x=0,  color="green",  linestyle="--", alpha=0.8, label="0 (identical)")
    ax.axvline(x=7,  color="orange", linestyle="--", alpha=0.7, label="7 days")
    ax.axvline(x=30, color="red",    linestyle="--", alpha=0.6, label="30 days")
    ax.set_xlabel("Std of raw D1 per group [capped at 100]")
    ax.set_ylabel("Number of groups")
    ax.set_title("Test A — Raw D1 Variability Within Group\n(naïve test)")
    ax.legend()

    ax = axes[1]
    ax.hist(anchor_fl["drift"].clip(upper=200), bins=100, color="#f4b4b4", alpha=0.9)
    ax.axvline(x=1,  color="green",  linestyle="--", alpha=0.8, label="1 day  (stable)")
    ax.axvline(x=7,  color="orange", linestyle="--", alpha=0.7, label="7 days")
    ax.axvline(x=30, color="red",    linestyle="--", alpha=0.6, label="30 days")
    ax.set_xlabel("|anchor_first − anchor_last| per group [capped at 200]")
    ax.set_ylabel("Number of groups")
    ax.set_title("Test B — Anchor Drift First vs Last Tx\n(correct stability test)")
    ax.legend()

    plt.tight_layout()
    plt.show()

    train = train.drop(columns=["D1_anchor"])
    return train


# ── 2.6 Other Dx Stability ───────────────────────────────────────────────────

def analyze_dx_stability(train: pd.DataFrame) -> None:
    """
    Section 2.6 — Stability table for D1/D2/D4/D10/D15 within card1+addr1 groups.
    Notebook cell: [94]
    """
    dx_check = [c for c in ["D1", "D2", "D4", "D10", "D15"] if c in train.columns]
    print(f"{'Col':<5}  {'NaN%':>5}  {'Med raw std':>12}  {'%std=0':>7}  "
          f"{'%std<1':>7}  {'%std<7':>7}  Verdict")
    print("-" * 72)
    for col in dx_check:
        grp_std = (train.dropna(subset=[col, "card1", "addr1"])
                   .groupby(["card1", "addr1"])[col]
                   .std().dropna())
        nan_p   = train[col].isna().mean() * 100
        med_std = grp_std.median()
        p0  = (grp_std == 0).mean() * 100
        p1  = (grp_std < 1).mean() * 100
        p7  = (grp_std < 7).mean() * 100
        v   = "✅ stable" if p1 > 80 else ("⚠️  partial" if p1 > 50 else "❌ dynamic")
        print(f"  {col:<3}  {nan_p:>5.1f}%  {med_std:>12.2f}  "
              f"{p0:>6.1f}%  {p1:>6.1f}%  {p7:>6.1f}%  {v}")
    print()
    print("Stable  → valid UID component OR normalized anchor (TransactionDT_days - Dx)")
    print("Dynamic → use as raw feature only, NOT in UID key")


# ── 2.7 UID Comparison ───────────────────────────────────────────────────────

def analyze_uid_d1_comparison(train: pd.DataFrame) -> None:
    """
    Section 2.7 — Compare card1+addr1 vs card1+addr1+D1: group fragmentation
    and fraud purity.
    Notebook cell: [97]
    """
    uid_configs = {
        "card1+addr1":    ["card1", "addr1"],
        "card1+addr1+D1": ["card1", "addr1", "D1"],
    }
    print(f"{'Metric':<40}  {'card1+addr1':>20}  {'card1+addr1+D1':>20}")
    print("-" * 84)

    stats = {}
    for lbl, cols in uid_configs.items():
        df    = train.dropna(subset=cols)
        grp   = df.groupby(cols)
        sizes = grp.size()
        frate = grp["isFraud"].mean()
        stats[lbl] = {"sizes": sizes, "frate": frate}

    for metric, fn in [
        ("Total groups",            lambda s: f"{len(s['sizes']):,}"),
        ("Avg tx/group",            lambda s: f"{s['sizes'].mean():.2f}"),
        ("Median tx/group",         lambda s: f"{s['sizes'].median():.0f}"),
        ("% groups with 1 tx",      lambda s: f"{(s['sizes']==1).mean():.1%}"),
        ("% groups with >=5 tx",    lambda s: f"{(s['sizes']>=5).mean():.1%}"),
        ("fraud_rate variance",     lambda s: f"{s['frate'].var():.6f}"),
        ("% groups fraud_rate = 1", lambda s: f"{(s['frate']==1.0).mean():.2%}"),
    ]:
        r1 = fn(stats["card1+addr1"])
        r2 = fn(stats["card1+addr1+D1"])
        print(f"  {metric:<38}  {r1:>20}  {r2:>20}")

    colors = {"card1+addr1": "#b3cde3", "card1+addr1+D1": "#f4b4b4"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for lbl, s in stats.items():
        ax.hist(s["sizes"].clip(upper=30), bins=30, alpha=0.6,
                color=colors[lbl], label=lbl, density=True)
    ax.axvline(x=1, color="red",    linestyle="--", alpha=0.5, label="1 tx")
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.5, label="5 tx")
    ax.set_xlabel("Group size (capped at 30)")
    ax.set_ylabel("Density")
    ax.set_title("Adding D1 — does it fragment groups?")
    ax.legend()

    ax = axes[1]
    for lbl, s in stats.items():
        ax.hist(s["frate"], bins=30, alpha=0.6,
                color=colors[lbl], label=lbl, density=True)
    ax.set_xlabel("Fraud rate per group")
    ax.set_ylabel("Density")
    ax.set_title("Adding D1 — does it improve fraud purity?")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ── 2.8 UID Anchor Fix — TransactionDT − D1 ─────────────────────────────────

def analyze_uid_anchor_fix(train: pd.DataFrame) -> None:
    """
    Section 2.8 — Compare card1+addr1 vs card1+addr1+round(anchor)
    where anchor = TransactionDT/86400 - D1.

    Tests whether the anchor transformation resolves card identity
    collisions within existing UID groups, improving fraud purity
    without the catastrophic fragmentation caused by raw D1.

    The anchor represents the approximate "card activation date":
    if two transactions share the same card1+addr1 but have different
    anchors, they likely belong to different physical cards (identity
    collision). Rounding to nearest day absorbs floating-point noise.

    Notebook cell: [new — between 2.7 and 2.9]
    """
    df = train.dropna(subset=["card1", "addr1", "D1"]).copy()
    
    # TransactionDT is in seconds; /86400 converts to days (D1 is already in days)
    df["anchor"] = (df["TransactionDT"] / 86400) - df["D1"]
    df["anchor_round"] = np.floor(df["anchor"])

    uid_configs = {
        "card1+addr1":              ["card1", "addr1"],
        "card1+addr1+anchor_round": ["card1", "addr1", "anchor_round"],
    }

    stats = {}
    for lbl, cols in uid_configs.items():
        grp   = df.groupby(cols)
        sizes = grp.size()
        frate = grp["isFraud"].mean()
        stats[lbl] = {"sizes": sizes, "frate": frate}

    print(f"{'Metric':<40}  {'card1+addr1':>20}  {'+ anchor_round':>20}")
    print("-" * 84)
    for metric, fn in [
        ("Total groups",            lambda s: f"{len(s['sizes']):,}"),
        ("Avg tx/group",            lambda s: f"{s['sizes'].mean():.2f}"),
        ("Median tx/group",         lambda s: f"{s['sizes'].median():.0f}"),
        ("% groups with 1 tx",      lambda s: f"{(s['sizes']==1).mean():.1%}"),
        ("% groups with >=5 tx",    lambda s: f"{(s['sizes']>=5).mean():.1%}"),
        ("fraud_rate variance",     lambda s: f"{s['frate'].var():.6f}"),
        ("% groups fraud_rate = 0", lambda s: f"{(s['frate']==0).mean():.2%}"),
        ("% groups fraud_rate = 1", lambda s: f"{(s['frate']==1.0).mean():.2%}"),
    ]:
        r1 = fn(stats["card1+addr1"])
        r2 = fn(stats["card1+addr1+anchor_round"])
        print(f"  {metric:<38}  {r1:>20}  {r2:>20}")

    # Collision analysis: how many groups get split?
    n_orig   = len(stats["card1+addr1"]["sizes"])
    n_anchor = len(stats["card1+addr1+anchor_round"]["sizes"])
    n_split  = n_anchor - n_orig
    pct_split = n_split / n_orig * 100 if n_orig > 0 else 0

    print(f"\n  Groups split by anchor   : {n_split:,} ({pct_split:.1f}% of original groups)")
    print(f"  Expansion factor         : {n_anchor / n_orig:.2f}×")

    # Compare fragmentation vs raw D1 (from Section 2.7)
    print(f"\n  Fragmentation comparison:")
    print(f"    card1+addr1+D1 (raw)     : 235,027 groups, 73.9% single-tx  ← catastrophic")
    print(f"    card1+addr1+anchor_round : {n_anchor:,} groups, "
          f"{(stats['card1+addr1+anchor_round']['sizes'] == 1).mean():.1%} single-tx")

    # Anchor uniqueness within groups: how many distinct anchors per group?
    anchors_per_group = df.groupby(["card1", "addr1"])["anchor_round"].nunique()
    multi_anchor = (anchors_per_group > 1).sum()
    print(f"\n  Groups with >1 distinct anchor (identity collisions): "
          f"{multi_anchor:,} ({multi_anchor / n_orig:.1%})")
    print(f"  Anchor values per group — median: {anchors_per_group.median():.0f}, "
          f"mean: {anchors_per_group.mean():.2f}, max: {anchors_per_group.max()}")

    # Visualize
    colors = {"card1+addr1": "#b3cde3", "card1+addr1+anchor_round": "#c4b7d4"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    ax = axes[0]
    for lbl, s in stats.items():
        ax.hist(s["sizes"].clip(upper=30), bins=30, alpha=0.6,
                color=colors[lbl], label=lbl, density=True)
    ax.axvline(x=1, color="red",    linestyle="--", alpha=0.5, label="1 tx")
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.5, label="5 tx")
    ax.set_xlabel("Group size (capped at 30)")
    ax.set_ylabel("Density")
    ax.set_title("Anchor Fix — Group Size Distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    for lbl, s in stats.items():
        ax.hist(s["frate"], bins=30, alpha=0.6,
                color=colors[lbl], label=lbl, density=True)
    ax.set_xlabel("Fraud rate per group")
    ax.set_ylabel("Density")
    ax.set_title("Anchor Fix — Fraud Purity")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(anchors_per_group.clip(upper=20), bins=20, color="#ffe0b2", alpha=0.9)
    ax.axvline(x=1, color="green", linestyle="--", alpha=0.7, label="1 anchor (no collision)")
    ax.set_xlabel("Distinct anchors per group (capped at 20)")
    ax.set_ylabel("Number of groups")
    ax.set_title("Identity Collisions per Group\n(>1 = multiple cards share same UID)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── 2.9 Amount Variance Per Group ────────────────────────────────────────────

def analyze_amount_variance_by_group(train: pd.DataFrame) -> None:
    """
    Section 2.9 — Amount std in fraud vs clean groups; amt_ratio distribution.
    Notebook cell: [101]
    """
    grp = (train.dropna(subset=["card1", "addr1"])
           .groupby(["card1", "addr1"]))

    amt_stats = grp["TransactionAmt"].agg(["std", "mean", "count"])
    amt_stats.columns = ["amt_std", "amt_mean", "tx_count"]
    amt_stats = amt_stats[amt_stats["tx_count"] >= 2].copy()
    fraud_flag = grp["isFraud"].max()
    amt_stats  = amt_stats.join(fraud_flag.rename("has_fraud"))

    clean_std = amt_stats[amt_stats["has_fraud"] == 0]["amt_std"].dropna()
    fraud_std = amt_stats[amt_stats["has_fraud"] == 1]["amt_std"].dropna()

    print("Amount std per group — Fraud vs Clean:")
    print(f"  {'':30}  {'Clean groups':>15}  {'Fraud groups':>15}")
    print("-" * 64)
    for label, fn in [
        ("Groups analyzed",      lambda s: f"{len(s):,}"),
        ("Median amt_std ($)",   lambda s: f"{s.median():.2f}"),
        ("Mean amt_std ($)",     lambda s: f"{s.mean():.2f}"),
        ("% groups std < $10",   lambda s: f"{(s<10).mean():.1%}"),
        ("% groups std > $100",  lambda s: f"{(s>100).mean():.1%}"),
        ("% groups std > $500",  lambda s: f"{(s>500).mean():.1%}"),
    ]:
        print(f"  {label:<30}  {fn(clean_std):>15}  {fn(fraud_std):>15}")

    fraud_grp_ids = amt_stats[amt_stats["has_fraud"] == 1].index
    in_fraud_grp  = (train.dropna(subset=["card1", "addr1"])
                     .set_index(["card1", "addr1"])
                     .loc[fraud_grp_ids]
                     .reset_index())
    print()
    print("Amount stats per transaction — Fraud vs Legitimate (within fraud groups):")
    for label, subset in [
        ("Legitimate tx in fraud groups", in_fraud_grp[in_fraud_grp["isFraud"] == 0]),
        ("Fraud tx in fraud groups",      in_fraud_grp[in_fraud_grp["isFraud"] == 1]),
    ]:
        m = subset["TransactionAmt"]
        print(f"  {label}:  median=${m.median():.0f}  mean=${m.mean():.0f}  "
              f"std=${m.std():.0f}  n={len(m):,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(clean_std.clip(upper=500), bins=80, alpha=0.6, color="#b3cde3", density=True,
            label=f"Clean groups (n={len(clean_std):,})")
    ax.hist(fraud_std.clip(upper=500), bins=80, alpha=0.7, color="#f4b4b4", density=True,
            label=f"Fraud groups (n={len(fraud_std):,})")
    ax.axvline(clean_std.median(), color="#1F4E79", linestyle="--", linewidth=1.5,
               label=f"Clean median ${clean_std.median():.0f}")
    ax.axvline(fraud_std.median(), color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Fraud median ${fraud_std.median():.0f}")
    ax.set_xlabel("std(TransactionAmt) per group [capped at $500]")
    ax.set_ylabel("Density")
    ax.set_title("Amount Variability Within Group\nFraud groups vs Clean groups")
    ax.legend()

    ax = axes[1]
    in_fraud_grp_full = in_fraud_grp.copy()
    group_means = (in_fraud_grp_full.groupby(["card1", "addr1"])["TransactionAmt"]
                   .transform("mean"))
    in_fraud_grp_full["amt_ratio"] = (in_fraud_grp_full["TransactionAmt"]
                                      / group_means.clip(lower=1))
    for label, mask, color in [
        ("Legitimate", in_fraud_grp_full["isFraud"] == 0, "#b3cde3"),
        ("Fraud",      in_fraud_grp_full["isFraud"] == 1, "#f4b4b4"),
    ]:
        ax.hist(in_fraud_grp_full[mask]["amt_ratio"].clip(0, 5),
                bins=60, alpha=0.65, color=color, density=True, label=label)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="ratio = 1 (= group mean)")
    ax.set_xlabel("TransactionAmt / group_mean [capped at 5×]")
    ax.set_ylabel("Density")
    ax.set_title("Amount Ratio: Fraud vs Legitimate\nWithin fraud groups only")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ── 2.10 Transaction Velocity ─────────────────────────────────────────────────

def analyze_velocity(train: pd.DataFrame) -> None:
    """
    Section 2.10 — Rolling velocity 3d/7d/30d: print fraud vs legit stats
    and show distributions. Uses closed='left' to exclude current transaction.
    Notebook cell: [104]
    """
    WINDOWS = {"3d": 3, "7d": 7, "30d": 30}

    train_sorted = (train.dropna(subset=["card1", "addr1"])
                    .sort_values("TransactionDT")
                    .copy())
    train_sorted["tx_day"] = train_sorted["TransactionDT"] / 86400

    for name, days in WINDOWS.items():
        col = f"velocity_{name}"
        train_sorted[col] = (
            train_sorted
            .groupby(["card1", "addr1"])["tx_day"]
            .transform(
                lambda x: x.rolling(window=days, min_periods=1, closed="left").count()
            )
        )

    print("Velocity distributions — Fraud vs Legitimate:")
    print(f"  {'Metric':<35}  {'Legitimate':>12}  {'Fraud':>12}  {'Fraud/Legit':>12}")
    print("-" * 76)
    legit = train_sorted[train_sorted["isFraud"] == 0]
    fraud = train_sorted[train_sorted["isFraud"] == 1]

    for col, name in [("velocity_3d",  "Count last 3d"),
                       ("velocity_7d",  "Count last 7d"),
                       ("velocity_30d", "Count last 30d")]:
        l_med  = legit[col].median()
        f_med  = fraud[col].median()
        l_mean = legit[col].mean()
        f_mean = fraud[col].mean()
        print(f"  {name} — median:          {l_med:>12.1f}  {f_med:>12.1f}  "
              f"{f_med/max(l_med, 0.01):>11.2f}x")
        print(f"  {name} — mean:            {l_mean:>12.2f}  {f_mean:>12.2f}  "
              f"{f_mean/max(l_mean, 0.01):>11.2f}x")
        print(f"  {name} — % with 0 prior:  "
              f"{(legit[col]==0).mean():>11.1%}  {(fraud[col]==0).mean():>11.1%}")
        print()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (col, title) in zip(axes, [
        ("velocity_3d",  "Velocity — Last 3 Days"),
        ("velocity_7d",  "Velocity — Last 7 Days"),
        ("velocity_30d", "Velocity — Last 30 Days"),
    ]):
        cap = train_sorted[col].quantile(0.99)
        for label, df, color in [("Legitimate", legit, "#b3cde3"), ("Fraud", fraud, "#f4b4b4")]:
            ax.hist(df[col].clip(upper=cap), bins=40, alpha=0.65,
                    color=color, density=True, label=label)
        ax.axvline(legit[col].median(), color="#1F4E79", linestyle="--", linewidth=1.5,
                   label=f"Legit med={legit[col].median():.0f}")
        ax.axvline(fraud[col].median(), color="#c0392b", linestyle="--", linewidth=1.5,
                   label=f"Fraud med={fraud[col].median():.0f}")
        ax.set_xlabel(f"Tx count in window [capped at p99={cap:.0f}]")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── 2.11 Device / Email Novelty ──────────────────────────────────────────────

def analyze_novelty(train: pd.DataFrame) -> None:
    """
    Section 2.11 — is_new_device / is_new_P_email / is_new_R_email: lift table
    and fraud rate bar charts.
    Notebook cell: [107]
    """
    novelty_cols = {
        "DeviceInfo":    "is_new_device",
        "P_emaildomain": "is_new_P_email",
        "R_emaildomain": "is_new_R_email",
    }

    train_sorted2 = (train.dropna(subset=["card1", "addr1"])
                     .sort_values("TransactionDT")
                     .copy())

    def _mark_new(series: pd.Series) -> list:
        seen, result = set(), []
        for val in series:
            if pd.isna(val):
                result.append(0)
            elif val in seen:
                result.append(0)
            else:
                result.append(1)
                seen.add(val)
        return result

    for src_col, new_col in novelty_cols.items():
        train_sorted2[new_col] = (
            train_sorted2.groupby(["card1", "addr1"])[src_col]
            .transform(_mark_new)
        )

    print("Novelty signal — Fraud rate when value is NEW vs SEEN BEFORE:")
    print(f"  {'Feature':<22}  {'% tx is_new':>12}  {'Fraud rate (new)':>18}  "
          f"{'Fraud rate (seen)':>18}  {'Lift':>8}")
    print("-" * 84)
    for src_col, new_col in novelty_cols.items():
        new_mask  = train_sorted2[new_col] == 1
        seen_mask = train_sorted2[new_col] == 0
        pct_new   = new_mask.mean()
        fr_new    = train_sorted2[new_mask]["isFraud"].mean()
        fr_seen   = train_sorted2[seen_mask]["isFraud"].mean()
        lift      = fr_new / max(fr_seen, 0.0001)
        print(f"  {new_col:<22}  {pct_new:>12.1%}  {fr_new:>18.3%}  "
              f"{fr_seen:>18.3%}  {lift:>7.2f}x")
    print()
    print("Note: 'Lift' = fraud_rate(new) / fraud_rate(seen before)")
    print("      Lift > 1 → new device/email is a fraud signal")
    print("      First transaction in any group always gets is_new=1 for all fields")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (src_col, new_col) in zip(axes, novelty_cols.items()):
        new_fr  = train_sorted2[train_sorted2[new_col] == 1]["isFraud"].mean()
        seen_fr = train_sorted2[train_sorted2[new_col] == 0]["isFraud"].mean()
        avg_fr  = train["isFraud"].mean()
        bars = ax.bar(["New (first time)", "Seen before"], [new_fr, seen_fr],
                      color=["#f4b4b4", "#b3cde3"], alpha=0.9, width=0.5)
        ax.axhline(avg_fr, color="black", linestyle="--", alpha=0.5,
                   label=f"Dataset avg {avg_fr:.3f}")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylabel("Fraud rate")
        ax.set_title(f"{new_col} Fraud Rate: New vs Seen Before")
        ax.set_ylim(0, max(new_fr, seen_fr) * 1.35)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()