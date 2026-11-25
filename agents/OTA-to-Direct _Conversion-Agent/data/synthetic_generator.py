"""Synthetic PMS bookings (OTA), CDP profiles, ground-truth links and a 15% ghost holdout.

Outcome model (known to the generator only):
  Y = Bernoulli(p0 + T * tau)  -- 'books direct on a future stay within 12 months'
  T = 1 if guest receives the intervention, 0 if in the ghost holdout (randomised 15%, stratified by OTA).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, ROOT

GEN_DIR = ROOT / "data" / "generated"
FIRST = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda","David","Elizabeth","William","Barbara","Richard","Susan",
         "Joseph","Jessica","Thomas","Sarah","Charles","Karen","Daniel","Nancy","Matthew","Lisa","Anthony","Betty","Mark","Sandra","Steven",
         "Ashley","Paul","Emily","Andrew","Donna","Joshua","Michelle","Kenneth","Carol","Kevin","Amanda","Brian","Melissa","George","Deborah",
         "Priya","Arjun","Wei","Mei","Hiro","Yuki","Omar","Fatima","Lucas","Sofia","Mateo","Isabella","Noah","Liam","Olivia","Emma"]
LAST = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson",
        "Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis",
        "Robinson","Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green","Adams","Nelson","Baker","Hall",
        "Rivera","Campbell","Mitchell","Carter","Roberts","Patel","Sharma","Kumar","Singh","Chen","Wang","Tanaka","Sato","Khan","Ali","Silva","Costa"]
OTAS = ["Expedia", "Booking.com", "Agoda"]
OTA_COMM = {"Expedia": (0.18, 0.03), "Booking.com": (0.15, 0.025), "Agoda": (0.17, 0.03)}
COUNTRIES = ["US", "GB", "DE", "IN", "AU", "SG", "FR", "JP"]
NICK = {"James": "Jim", "Robert": "Bob", "William": "Bill", "Richard": "Rick", "Elizabeth": "Liz", "Michael": "Mike", "Patricia": "Pat",
        "Jennifer": "Jen", "Thomas": "Tom", "Charles": "Chuck", "Daniel": "Dan", "Matthew": "Matt", "Anthony": "Tony", "Joseph": "Joe"}


def _typo(s: str, rng) -> str:
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 2)); k = int(rng.integers(0, 3))
    if k == 0:
        return s[:i] + s[i + 1:]                                   # deletion
    if k == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]                 # transposition
    return s[:i] + chr(ord("a") + int(rng.integers(0, 26))) + s[i + 1:]  # substitution


def generate(n_bookings: int | None = None, seed: int | None = None, save: bool = True):
    s = get_settings(); n = n_bookings or s.N_BOOKINGS; seed = s.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    # ---------------- CDP profiles ----------------
    n_prof = max(int(n * 0.30), 200)
    pid = np.array([f"P{i:06d}" for i in range(n_prof)])
    pf = rng.choice(FIRST, n_prof); pl = rng.choice(LAST, n_prof)
    phone = rng.integers(10**9, 10**10 - 1, n_prof)
    cdp = pd.DataFrame({
        "profile_id": pid, "first_name": pf, "last_name": pl,
        "email": [f"{a.lower()}.{b.lower()}{rng.integers(1, 999)}@mail.com" for a, b in zip(pf, pl)],
        "phone": phone.astype(str), "country": rng.choice(COUNTRIES, n_prof),
        "loyalty_tier": rng.choice(["none", "member", "silver", "gold"], n_prof, p=[.45, .30, .17, .08]),
        "past_direct_stays": rng.poisson(1.2, n_prof), "past_ota_stays": rng.poisson(2.0, n_prof),
        "lifetime_value_usd": np.round(rng.gamma(2.0, 600, n_prof), 2),
        "email_opt_in": rng.random(n_prof) < 0.7, "msgs_30d": rng.poisson(0.8, n_prof)})

    # ---------------- PMS bookings ----------------
    ota = rng.choice(OTAS, n, p=[.42, .38, .20])
    linked = rng.random(n) < 0.58
    link_idx = rng.integers(0, n_prof, n)
    true_pid = np.where(linked, pid[link_idx], None)
    base = cdp.iloc[link_idx].reset_index(drop=True)
    nf = rng.choice(FIRST, n); nl = rng.choice(LAST, n); nph = rng.integers(10**9, 10**10 - 1, n).astype(str); nc = rng.choice(COUNTRIES, n)
    fn = np.where(linked, base.first_name.values, nf); ln = np.where(linked, base.last_name.values, nl)
    ph = np.where(linked, base.phone.values, nph); ctry = np.where(linked, base.country.values, nc)
    noise = rng.random(n)
    fn = np.array([NICK.get(f, f) if (l and r < 0.10) else (_typo(f, rng) if (l and r < 0.16) else f) for f, l, r in zip(fn, linked, noise)])
    ln = np.array([_typo(x, rng) if (l and r > 0.93) else x for x, l, r in zip(ln, linked, noise)])
    phone_present = rng.random(n) < 0.90
    email_unmasked = linked & (rng.random(n) < 0.15)
    masked = np.array([f"{rng.integers(10**7, 10**8)}@m.{o.split('.')[0].lower()}.com" for o in ota])
    email = np.where(email_unmasked, base.email.values, masked)

    adr = np.round(np.clip(rng.lognormal(5.0, 0.35, n), 55, 900), 2)
    los = np.clip(rng.poisson(2.2, n) + 1, 1, 14)
    comm = np.array([np.clip(rng.normal(*OTA_COMM[o]), 0.10, 0.30) for o in ota]).round(4)
    lead = np.clip(rng.gamma(2.0, 18.0, n), 0, 330).astype(int)
    checkin = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
    purpose = rng.choice(["leisure", "business", "group"], n, p=[.55, .35, .10])
    free_cancel = rng.random(n) < np.where(ota == "Booking.com", 0.8, 0.55)

    # ---------------- ghost holdout: exactly 15% randomised, stratified by OTA ----------------
    holdout = np.zeros(n, dtype=bool)
    for o in OTAS:
        idx = np.where(ota == o)[0]; k = int(round(len(idx) * s.HOLDOUT_FRAC))
        holdout[rng.choice(idx, k, replace=False)] = True
    treated = ~holdout

    # ---------------- outcomes ----------------
    pds = np.where(linked, base.past_direct_stays.values, 0); tier = np.where(linked, base.loyalty_tier.values, "none")
    tier_n = pd.Series(tier).map({"none": 0, "member": 1, "silver": 2, "gold": 3}).values
    z0 = -2.0 + 0.35 * pds + 0.45 * tier_n + 0.0008 * adr + 0.25 * (purpose == "business") - 0.15 * (ota == "Agoda")
    p0 = 1 / (1 + np.exp(-z0))
    tau = (0.10 + 0.05 * (purpose == "leisure") - 0.04 * tier_n + 0.0002 * adr - 0.002 * los
           + 0.04 * (ota == "Expedia") - 0.08 * (pds >= 3))        # heterogeneous true uplift; some negative (sleeping dogs)
    tau = np.clip(tau, -0.15, 0.40)
    y = rng.random(n) < np.clip(p0 + treated * tau, 0.001, 0.999)
    zc = (-3.2 + 0.030 * lead + 1.5 * free_cancel + 0.6 * (ota == "Booking.com") - 0.6 * tier_n - 0.4 * (los > 3)
          + 0.6 * (purpose == "leisure") * (lead > 60) - 0.5 * (purpose == "business"))
    cancel = rng.random(n) < 1 / (1 + np.exp(-zc))

    bookings = pd.DataFrame({
        "booking_id": [f"B{i:07d}" for i in range(n)], "ota": ota, "masked_email": email, "email_is_real": email_unmasked,
        "first_name": fn, "last_name": ln, "phone_last4": np.where(phone_present, np.array([x[-4:] for x in ph]), None), "country": ctry,
        "checkin_date": checkin.strftime("%Y-%m-%d"), "los": los, "adr": adr, "ota_commission": comm, "lead_days": lead,
        "trip_purpose": purpose, "free_cancellation": free_cancel,
        "is_ghost_holdout": holdout, "treated": treated,
        "true_profile_id": true_pid,                                # ground truth (not available at inference)
        "true_tau": tau.round(4), "converted_direct": y, "cancelled": cancel})
    bookings["saved_commission_usd"] = (bookings.adr * bookings.los * bookings.ota_commission).round(2)
    if save:
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        bookings.to_csv(GEN_DIR / "pms_bookings.csv", index=False); cdp.to_csv(GEN_DIR / "cdp_profiles.csv", index=False)
    return bookings, cdp


def load():
    return pd.read_csv(GEN_DIR / "pms_bookings.csv", dtype={"phone_last4": str}), pd.read_csv(GEN_DIR / "cdp_profiles.csv", dtype={"phone": str})


if __name__ == "__main__":
    b, c = generate()
    print(b.shape, c.shape, "holdout:", round(b.is_ghost_holdout.mean(), 4), "linked:", round(b.true_profile_id.notna().mean(), 3))
