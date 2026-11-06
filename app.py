import math
import json
import uuid
import time
import random
from threading import Lock
from contextlib import nullcontext
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def closed_candles(raw, interval, now=None):
    x = raw.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x[["Open", "High", "Low", "Close", "Volume"]].dropna()
    x.index = pd.to_datetime(x.index, utc=True)
    x = x.sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    duration = pd.Timedelta(minutes=15) if interval == "15m" else pd.Timedelta(hours=1)
    # Yahoo timestamps mark candle starts. Allow a short finalization buffer.
    return x.loc[x.index + duration <= now - pd.Timedelta(seconds=60)]


def indicators(x):
    x = x.copy()
    for n in (9, 21, 50):
        x[f"EMA{n}"] = x.Close.ewm(span=n, adjust=False).mean()
    d = x.Close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    x["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    x.loc[(loss == 0) & (gain == 0), "RSI"] = 50
    previous = x.Close.shift(1)
    tr = pd.concat([x.High - x.Low, (x.High - previous).abs(),
                    (x.Low - previous).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    x["VOL20"] = x.Volume.rolling(20).mean()
    return x.iloc[50:].dropna()


def signal(r):
    if r.Volume <= r.VOL20:
        return "WAIT"
    if r.EMA9 > r.EMA21 > r.EMA50 and r.Close > r.EMA21 and 52 <= r.RSI <= 70:
        return "UP"
    if r.EMA9 < r.EMA21 < r.EMA50 and r.Close < r.EMA21 and 30 <= r.RSI <= 48:
        return "DOWN"
    return "WAIT"


def direction_test(data, interval, max_daily):
    duration = pd.Timedelta(minutes=15) if interval == "15m" else pd.Timedelta(hours=1)
    rows, counts = [], {}
    # Fixed rules: reserve latest 30% of candles for a chronological holdout.
    boundary = int(len(data) * 0.7)
    for i in range(len(data) - 1):
        r, nxt = data.iloc[i], data.iloc[i + 1]
        direction = signal(r)
        if direction == "WAIT" or data.index[i + 1] - data.index[i] != duration:
            continue
        ts = data.index[i + 1]
        day = str(ts.tz_convert("America/New_York").date())
        if counts.get(day, 0) >= max_daily:
            continue
        counts[day] = counts.get(day, 0) + 1
        move = float(nxt.Close - nxt.Open)
        outcome = "UP" if move > 0 else "DOWN" if move < 0 else "TIE"
        rows.append(dict(time=ts, sample="Holdout" if i + 1 >= boundary else "Earlier",
                         signal=direction, candle_open=float(nxt.Open),
                         candle_close=float(nxt.Close), outcome=outcome,
                         correct=direction == outcome))
    return pd.DataFrame(rows)


def wilson(wins, n):
    if not n:
        return None
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def paper_pnl(price_cents, quantity, entry_fee, mode, exit_cents=0, exit_fee=0):
    cost = quantity * price_cents / 100 + entry_fee
    proceeds = quantity * (1 if mode == "WIN" else 0 if mode == "LOSS" else exit_cents / 100)
    return cost, proceeds - (exit_fee if mode == "SELL" else 0) - cost


def features(data):
    # Each row uses only information available at that candle's end.
    scale = data.ATR.replace(0, np.nan)
    return pd.DataFrame({
        "trend_fast": (data.EMA9 - data.EMA21) / scale,
        "trend_slow": (data.EMA21 - data.EMA50) / scale,
        "distance": (data.Close - data.EMA21) / scale,
        "momentum": (data.RSI - 50) / 50,
        "return_1": data.Close.pct_change(),
        "return_4": data.Close.pct_change(4),
        "volume": np.log1p(data.Volume / data.VOL20.replace(0, np.nan)),
        "body": (data.Close - data.Open) / scale,
    }, index=data.index).replace([np.inf, -np.inf], np.nan)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def fit_model(x, y):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-9, 1, std)
    z = np.clip((x - mean) / std, -8, 8)
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(z.shape[1])
    # Fixed regularization and iterations; no fitting against validation or test.
    for _ in range(300):
        gradient = z.T @ (sigmoid(z @ w) - y) / len(y)
        gradient[1:] += 0.05 * w[1:]
        w -= 0.1 * gradient
    return mean, std, w


def predict(model, x):
    mean, std, w = model
    z = np.clip((x - mean) / std, -8, 8)
    return sigmoid(np.column_stack([np.ones(len(z)), z]) @ w)


def filter_predictions(frame, threshold, max_daily):
    if frame.empty:
        return frame.copy()
    eligible = frame.loc[np.maximum(frame.p_up, 1 - frame.p_up) >= threshold].copy()
    days = eligible.index.tz_convert("America/New_York").date
    return eligible.loc[eligible.groupby(days).cumcount() < max_daily]


def model_research(data, interval, max_daily):
    f = features(data)
    nxt = data.shift(-1)
    duration = pd.Timedelta(minutes=15) if interval == "15m" else pd.Timedelta(hours=1)
    contiguous = data.index.to_series().shift(-1) - data.index.to_series() == duration
    frame = f.copy()
    frame["y"] = (nxt.Close > nxt.Open).astype(float)
    # Ties have no directional winner in this proxy.
    frame = frame.loc[contiguous & (nxt.Close != nxt.Open)].dropna()
    if len(frame) < 300:
        return None
    a, b = int(len(frame) * 0.6), int(len(frame) * 0.8)
    train, validation, test = frame.iloc[:a - 1], frame.iloc[a:b - 1], frame.iloc[b:]
    if train.y.nunique() < 2:
        return None
    model = fit_model(train[f.columns].to_numpy(), train.y.to_numpy())
    def score(part):
        result = part[["y"]].copy()
        result["p_up"] = predict(model, part[f.columns].to_numpy())
        result["prediction"] = np.where(result.p_up >= 0.5, "UP", "DOWN")
        result["correct"] = (result.p_up >= 0.5) == (result.y == 1)
        return result
    val, test = score(validation), score(test)
    majority = int(train.y.mean() >= 0.5)
    baseline = float((val.y == majority).mean())
    candidates = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        chosen = filter_predictions(val, threshold, max_daily)
        n = len(chosen)
        lo, hi = wilson(int(chosen.correct.sum()), n) if n else (0, 1)
        candidates.append(dict(threshold=threshold, signals=n,
                               accuracy=chosen.correct.mean() if n else np.nan,
                               lower_95=lo, upper_95=hi))
    # Selection is exploratory. Only enable a policy with >=30 validation signals
    # and a lower confidence bound above the training-majority validation baseline.
    qualified = [r for r in candidates if r["signals"] >= 30 and r["lower_95"] > baseline]
    chosen = max(qualified, key=lambda r: r["lower_95"]) if qualified else None
    threshold = chosen["threshold"] if chosen else 0.60
    selected = filter_predictions(test, threshold, max_daily)
    last_features = f.iloc[[-1]].dropna()
    latest = float(predict(model, last_features.to_numpy())[0]) if len(last_features) else None
    return dict(train_n=len(train), validation=val, test=test, selected=selected,
                candidates=pd.DataFrame(candidates), threshold=threshold,
                validation_qualified=bool(chosen), latest=latest, majority=majority,
                base_probability=float(train.y.mean()))


def live_snapshot(report, candle_end, interval, max_daily, journal, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    duration = pd.Timedelta(minutes=15) if interval == "15m" else pd.Timedelta(hours=1)
    end = pd.Timestamp(candle_end)
    p = report["latest"] if report else None
    reason = "Not enough valid model data"
    action = "SKIP"
    # The training target is the candle immediately after this completed candle.
    # Do not present its original score as a prediction for a later horizon.
    today = now.tz_convert("America/New_York").date()
    used = sum(pd.Timestamp(r["opened_at"]).tz_convert("America/New_York").date() == today
               and r["interval"] == interval for r in journal)
    if now < end or now >= end + duration:
        reason = "The forecast window is unavailable or has already ended"
    elif now - end > pd.Timedelta(minutes=2):
        reason = "Late entry: refresh shortly after the next candle closes"
    elif used >= max_daily:
        reason = "Daily simulated entry limit reached for this timeframe"
    elif report and p is not None:
        selected = report["selected"]
        n = len(selected)
        lo = wilson(int(selected.correct.sum()), n)[0] if n else 0
        baseline = float((selected.y == report["majority"]).mean()) if n else 1
        test = report["test"]
        brier = float(((test.p_up - test.y) ** 2).mean())
        base_brier = float(((report["base_probability"] - test.y) ** 2).mean())
        supported = report["validation_qualified"] and n >= 30 and lo > baseline and brier < base_brier
        if not supported:
            reason = "Model has not passed the historical evidence gate"
        elif max(p, 1 - p) < report["threshold"]:
            reason = "Model estimate is below the selected confidence threshold"
        else:
            action = "UP" if p >= .5 else "DOWN"
            reason = "Experimental signal passed timing, sample size, and evidence checks"
    return dict(action=action, reason=reason, p_up=p, interval=interval,
                candle_end=end.isoformat(), window_end=(end + duration).isoformat(),
                generated_at=now.isoformat())


def show_model(data, interval, max_daily, stale):
    st.subheader("Selective model - test results")
    st.caption("Logistic regression estimates next-candle UP probability. Older 60% trains the model, "
               "next 20% selects a confidence filter, latest 20% tests it. One boundary row is purged at "
               "each split. The model is not a Kalshi settlement probability model.")
    with st.spinner("Fitting and evaluating the directional model..."):
        report = model_research(data, interval, max_daily)
    if report is None:
        st.warning("Not enough valid data to fit and test this model.")
        return
    test, selected = report["test"], report["selected"]
    n = len(selected)
    lo, hi = wilson(int(selected.correct.sum()), n) if n else (0, 1)
    cols = st.columns(4)
    cols[0].metric("Filtered test accuracy", f"{100 * selected.correct.mean():.1f}%" if n else "No signals")
    cols[1].metric("Filtered test signals", n)
    cols[2].metric("95% interval", f"{100 * lo:.1f} to {100 * hi:.1f}%" if n else "N/A")
    cols[3].metric("Test signal coverage", f"{100 * n / len(test):.1f}%")
    baseline = float((selected.y == report["majority"]).mean()) if n else 1
    all_accuracy = float(test.correct.mean())
    brier = float(((test.p_up - test.y) ** 2).mean())
    base_brier = float(((report["base_probability"] - test.y) ** 2).mean())
    st.caption(f"Unfiltered test accuracy: {100 * all_accuracy:.1f}%. Training-majority baseline "
               f"on selected test candles: {100 * baseline:.1f}%" if n else
               f"Unfiltered test accuracy: {100 * all_accuracy:.1f}%. No filtered test candles.")
    st.caption(f"Test Brier score: {brier:.3f}; constant training-probability baseline: "
               f"{base_brier:.3f} (lower is better). Filter threshold: {100 * report['threshold']:.0f}%. "
               "A model score is not an observed winning percentage.")
    supported = report["validation_qualified"] and n >= 30 and lo > baseline and brier < base_brier
    p = report["latest"]
    if supported and not stale and p is not None and max(p, 1 - p) >= report["threshold"]:
        st.info(f"Historical evidence gate passed for {'UP' if p >= 0.5 else 'DOWN'}. See the live panel for timing and entry limits.")
    else:
        st.warning("Model signal: WAIT. Confidence, freshness, sample size, or comparative test evidence "
                   "does not meet the research gate.")
    if p is not None:
        st.write(f"Latest model UP estimate: {100 * p:.1f}% | DOWN estimate: {100 * (1 - p):.1f}%")
    if not report["validation_qualified"]:
        st.caption("No candidate passed the validation gate. The 60% filter is shown for diagnosis only.")
    calibration = test.copy()
    calibration["score_band"] = pd.cut(calibration.p_up, bins=[0, .2, .4, .6, .8, 1], include_lowest=True)
    bins = calibration.groupby("score_band", observed=True).agg(
        count=("y", "size"), mean_model_up=("p_up", "mean"), observed_up=("y", "mean"))
    with st.expander("Validation thresholds and test probability reliability"):
        st.dataframe(report["candidates"], use_container_width=True)
        st.dataframe(bins, use_container_width=True)
        st.write("Five thresholds were compared; validation intervals do not correct for that selection. "
                 "Test observations can be correlated. Test results are retrospective and previously "
                 "viewed data is not a pristine holdout. Rechecking or changing settings against the same "
                 "test can overfit. Collect new forward paper observations before claiming improvement.")
    st.download_button("Download model test CSV", test.to_csv(), f"btc_model_test_{interval}.csv", "text/csv")
    return report


@st.cache_data(ttl=60)
def fetch(interval):
    return yf.download("BTC-USD", period="60d" if interval == "15m" else "730d",
                       interval=interval, auto_adjust=True, progress=False)


class FeedError(Exception):
    pass


class RateLimitError(FeedError):
    pass


@st.cache_resource
def kalshi_request_gate():
    # Shared across reruns and sessions in this Streamlit process.
    return {"lock": Lock(), "next_at": 0.0}


def get_json(url):
    is_kalshi = url.startswith("https://external-api.kalshi.com/trade-api/v2/")
    gate = kalshi_request_gate() if is_kalshi else None
    attempts = 5 if is_kalshi else 1
    with gate["lock"] if gate else nullcontext():
        for attempt in range(attempts):
            if gate:
                time.sleep(max(0.0, gate["next_at"] - time.monotonic()))
                gate["next_at"] = time.monotonic() + 1.0
            try:
                with urlopen(Request(url, headers={"User-Agent": "SignalLab-PaperResearch/7"}), timeout=20) as response:
                    body = json.load(response)
                    quota = response.headers.get("x-requests-remaining", "unknown")
                return body, quota, datetime.now(timezone.utc).isoformat()
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if is_kalshi and code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                    gate["next_at"] = time.monotonic() + 2 ** (attempt + 1) + random.uniform(0, 0.5)
                    continue
                # Never expose request URLs or bodies (sports URLs can contain keys).
                if code == 429 and is_kalshi:
                    gate["next_at"] = time.monotonic() + 60
                    raise RateLimitError("Kalshi rate limit persists after retries. Wait at least 60 seconds before rescanning.") from None
                messages = {401: "API key rejected", 403: "Access unavailable for this plan",
                            404: "Market or event not found", 429: "Quota or rate limit reached"}
                raise FeedError(messages.get(code, f"Provider returned HTTP {code}")) from None
            except (URLError, TimeoutError, OSError):
                if is_kalshi and attempt < attempts - 1:
                    gate["next_at"] = time.monotonic() + 2 ** (attempt + 1) + random.uniform(0, 0.5)
                    continue
                raise FeedError("Feed unavailable. Retry later.") from None
            except ValueError:
                raise FeedError("Feed returned an invalid response. Retry later.") from None


@st.cache_data(ttl=300, show_spinner=False)
def sports_feed(sport, api_key, markets="h2h,spreads,totals", event_id="", scores=False):
    root = f"https://api.the-odds-api.com/v4/sports/{quote(sport, safe='')}"
    params = {"apiKey": api_key}
    if scores:
        path = "/scores"
        params.update(daysFrom=3, dateFormat="iso")
    else:
        path = f"/events/{quote(event_id, safe='')}/odds" if event_id else "/odds"
        params.update(regions="us", markets=markets, oddsFormat="decimal", dateFormat="iso")
    return get_json(root + path + "?" + urlencode(params))


@st.cache_data(ttl=30, show_spinner=False)
def kalshi_feed(ticker):
    return get_json("https://external-api.kalshi.com/trade-api/v2/markets/" + quote(ticker, safe=""))


def odds_rows(events, fetched):
    rows = []
    for event in events:
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                groups = {}
                for item in market.get("outcomes", []):
                    price = item.get("price")
                    if not isinstance(price, (int, float)) or not np.isfinite(price) or price <= 1:
                        continue
                    point = item.get("point")
                    # Spreads have opposite signed points; pair the exact absolute line.
                    group_line = abs(point) if market["key"] == "spreads" and point is not None else point
                    group = (item.get("description", ""), group_line)
                    groups.setdefault(group, []).append(item)
                for group, outcomes in groups.items():
                    valid_pair = len(outcomes) == 2 or (market["key"] == "h2h" and len(outcomes) == 3)
                    names = [o["name"] for o in outcomes]
                    valid_pair = valid_pair and len(set(names)) == len(names)
                    if market["key"] == "spreads" and valid_pair:
                        valid_pair = abs(sum(float(o.get("point", 0)) for o in outcomes)) < 1e-8
                    denom = sum(1 / o["price"] for o in outcomes) if valid_pair else None
                    updated = market.get("last_update", book.get("last_update", ""))
                    stamp = pd.to_datetime(updated, utc=True, errors="coerce")
                    age = (pd.Timestamp(fetched) - stamp).total_seconds() / 60 if pd.notna(stamp) else np.nan
                    for outcome in outcomes:
                        d = float(outcome["price"])
                        rows.append(dict(event_id=event["id"], sport=event.get("sport_key", ""),
                            game=f"{event.get('away_team', '')} @ {event.get('home_team', '')}",
                            start=event.get("commence_time", ""), book=book["title"],
                            market=market["key"], player=outcome.get("description", ""),
                            selection=outcome["name"], line=outcome.get("point"), decimal_odds=d,
                            american_odds=round((d - 1) * 100 if d >= 2 else -100 / (d - 1)),
                            implied_pct=100 / d,
                            normalized_market_pct=100 / d / denom if denom else np.nan,
                            quote_time=updated, quote_age_minutes=age, fetched_at=fetched))
    return pd.DataFrame(rows)


def best_quotes(rows):
    if rows.empty:
        return rows.copy()
    return rows.sort_values("decimal_odds", ascending=False).drop_duplicates(
        ["event_id", "market", "player", "selection", "line"])


def sports_profit(stake, decimal_odds, result):
    if result == "WIN":
        return stake * (decimal_odds - 1)
    if result == "LOSS":
        return -stake
    if result in ("PUSH", "VOID"):
        return 0.0
    return None


def game_result(entry, event):
    if not event.get("completed") or not event.get("scores"):
        return "PENDING"
    scores = {r["name"]: float(r["score"]) for r in event["scores"]}
    home, away = event.get("home_team"), event.get("away_team")
    if home not in scores or away not in scores:
        return "PENDING"
    if entry["market"] in ("h2h", "spreads"):
        team = entry["selection"]
        if team not in (home, away):
            return "MANUAL"
        other = away if team == home else home
        margin = scores[team] - scores[other]
        if entry["market"] == "spreads":
            margin += float(entry["line"])
    elif entry["market"] == "totals":
        total = scores[home] + scores[away]
        margin = total - float(entry["line"])
        if entry["selection"] == "Under":
            margin = -margin
        elif entry["selection"] != "Over":
            return "MANUAL"
    else:
        return "MANUAL"
    return "WIN" if margin > 0 else "LOSS" if margin < 0 else "PUSH"


def show_sports_journal(api_key):
    st.subheader("Sports paper journal")
    st.caption("Simulated singles only. Export before closing; session storage is temporary. "
               "Import your CSV to resume. No real bets are placed.")
    journal = st.session_state.setdefault("sports_journal", [])
    uploaded = st.file_uploader("Import saved sports paper journal", type="csv")
    if uploaded and st.button("Import journal CSV"):
        try:
            loaded = pd.read_csv(uploaded)
            required = {"event_id", "sport", "game", "market", "selection", "line", "decimal_odds", "stake", "result"}
            if not required.issubset(loaded.columns) or len(loaded) > 10000:
                raise ValueError()
            if not pd.to_numeric(loaded.stake).ge(0).all() or not pd.to_numeric(loaded.decimal_odds).gt(1).all():
                raise ValueError()
            if not loaded.result.isin(["PENDING", "MANUAL", "WIN", "LOSS", "PUSH", "VOID"]).all():
                raise ValueError()
            journal[:] = loaded.to_dict("records")
            st.success("Journal imported.")
        except Exception:
            st.error("Invalid journal CSV. Use a journal exported by this app.")
    if not journal:
        st.write("No paper entries yet.")
        return
    if st.button("Fetch final scores and settle game-line entries"):
        for sport in sorted({row["sport"] for row in journal if row["result"] == "PENDING"}):
            if sport not in ("americanfootball_nfl", "baseball_mlb"):
                continue
            try:
                events, quota, _ = sports_feed(sport, api_key, scores=True)
                lookup = {e["id"]: e for e in events}
                for entry in journal:
                    if entry["sport"] == sport and entry["result"] == "PENDING" and entry["event_id"] in lookup:
                        entry["result"] = game_result(entry, lookup[entry["event_id"]])
                st.caption(f"{sport}: credits remaining {quota}")
            except FeedError as exc:
                st.error(str(exc))
    index = st.selectbox("Entry to settle or correct manually", range(len(journal)),
                         format_func=lambda i: f"{i + 1}: {journal[i]['game']} | {journal[i]['selection']}")
    result = st.selectbox("Observed result - confirm sportsbook rules", ["PENDING", "WIN", "LOSS", "PUSH", "VOID"])
    if st.button("Save observed paper result"):
        journal[index]["result"] = result
    frame = pd.DataFrame(journal)
    frame["net_pnl"] = [sports_profit(float(r['stake']), float(r['decimal_odds']), r['result']) for r in journal]
    settled = frame[frame.result.isin(["WIN", "LOSS", "PUSH", "VOID"])]
    decisive = settled[settled.result.isin(["WIN", "LOSS"])]
    cols = st.columns(3)
    cols[0].metric("Settled paper entries", len(settled))
    cols[1].metric("Paper net P/L", f"${settled.net_pnl.sum():+.2f}")
    cols[2].metric("Paper ROI", f"{100 * settled.net_pnl.sum() / settled.stake.sum():.1f}%" if settled.stake.sum() else "N/A")
    if len(decisive):
        wins = int((decisive.result == "WIN").sum())
        lo, hi = wilson(wins, len(decisive))
        st.caption(f"Win rate: {100 * wins / len(decisive):.1f}% from {len(decisive)} decisions; "
                   f"95% interval {100 * lo:.1f} to {100 * hi:.1f}%. Overlapping bets can be correlated.")
    st.dataframe(frame, use_container_width=True)
    st.download_button("Save sports paper journal CSV", frame.to_csv(index=False), "sports_paper_journal.csv", "text/csv")
    st.caption("Automatic settlement uses final game scores available for the last three days. "
               "Player props, postponed games, special rules and voids need manual review.")


def show_sports(sport_name="NFL", sport="americanfootball_nfl", markets="h2h,spreads,totals"):
    st.header(f"{sport_name} • Odds research")
    try:
        api_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        st.info("Automatic sports data needs your Odds API key. Add ODDS_API_KEY in Streamlit's Secrets settings.")
        st.code('ODDS_API_KEY = "YOUR_KEY"', language="toml")
        st.link_button("Get an Odds API key", "https://the-odds-api.com/")
        st.caption("Do not put your key in GitHub or chat. If this app is public, visitors can use "
                   "its shared feed quota. Make it private before adding your key.")
        return
    st.caption("Feeds fetch automatically on opening this section. Responses are cached for five minutes. "
               "There is no background polling while the app is closed.")
    if st.button("Refresh sports feeds"):
        sports_feed.clear()
    try:
        with st.spinner("Loading game odds..."):
            events, quota, fetched = sports_feed(sport, api_key, markets)
    except FeedError as exc:
        st.error(str(exc))
        return
    st.caption(f"The Odds API | Fetched {fetched} | Credits remaining at fetch: {quota}. "
               "Game lines request up to 3 credits; props add credits; final scores request up to 2 per sport.")
    mode = st.radio("Markets", ["Game lines", "Player props"] if sport in ("americanfootball_nfl", "baseball_mlb") else ["Game lines"], horizontal=True)
    if mode == "Player props" and events:
        event_map = {e["id"]: e for e in events}
        selected_event = st.selectbox("Game", list(event_map), format_func=lambda k:
            f"{event_map[k]['away_team']} @ {event_map[k]['home_team']}")
        options = {"Passing yards": "player_pass_yds", "Rushing yards": "player_rush_yds",
                   "Receiving yards": "player_reception_yds", "Receptions": "player_receptions"} if sport == "americanfootball_nfl" else {
                   "Pitcher strikeouts": "pitcher_strikeouts", "Hits": "batter_hits", "Total bases": "batter_total_bases"}
        label = st.selectbox("Prop category", list(options))
        try:
            with st.spinner("Loading available player props..."):
                event, quota, fetched = sports_feed(sport, api_key, options[label], selected_event)
            events_for_rows = [event]
            st.caption(f"Prop feed credits remaining: {quota}")
        except FeedError as exc:
            st.error(str(exc))
            events_for_rows = []
    else:
        events_for_rows = events
    rows = odds_rows(events_for_rows, fetched)
    st.info("These percentages come from sportsbook prices. They are market estimates, not a trained "
            "sports prediction model or guaranteed win rates. Player availability is not verified by this feed.")
    if rows.empty:
        st.write("No available odds for this selection. Markets may not be listed yet.")
    else:
        books = st.multiselect("Books to compare", sorted(rows.book.unique()), default=sorted(rows.book.unique()))
        view = rows[rows.book.isin(books)].copy()
        fresh = view[view.quote_age_minutes.between(0, 10)].copy()
        if len(fresh) < len(view):
            st.warning("Some quotes are older than 10 minutes or lack timestamps. Those are excluded from price comparisons.")
        st.subheader("Best available price for each exact selection")
        best = best_quotes(fresh)
        columns = ["game", "book", "market", "player", "selection", "line", "american_odds",
                   "implied_pct", "normalized_market_pct", "quote_time"]
        simple_pick_cards(eligible_quotes(best))
        with st.expander("Full comparison table"):
            st.dataframe(best[columns] if not best.empty else best, use_container_width=True)
        st.caption("Implied % = 1 / decimal odds. Normalized market % proportionally removes the book's "
                   "margin only where a complete outcome pair exists. For lines with a possible push, "
                   "this is conditional on a decisive result. Higher payout alone does not establish value.")
        with st.expander("All quotes"):
            st.dataframe(view[columns], use_container_width=True)
        st.download_button("Export current odds snapshot", view.to_csv(index=False),
                           f"{sport_name.lower()}_odds.csv", "text/csv")
        pregame = best.loc[pd.to_datetime(best.start, utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC")] if not best.empty else best
        render_parlays(fresh)
        st.subheader("Record a paper single")
        st.caption("Only fresh pregame quotes can be logged. Entries are simulations with a $0 real-money budget.")
        if not pregame.empty:
            pregame = pregame.reset_index(drop=True)
            choice = st.selectbox("Paper selection", range(len(pregame)), format_func=lambda i:
                f"{pregame.iloc[i].game} | {pregame.iloc[i].player} {pregame.iloc[i].selection} "
                f"{pregame.iloc[i].line} | {pregame.iloc[i].book} {pregame.iloc[i].american_odds:+}")
            stake = st.number_input("Virtual single stake ($)", 0.0, 10.0, 1.0, 0.5)
            if st.button("Log simulated single", disabled=stake <= 0):
                row = pregame.iloc[choice].to_dict()
                start = pd.to_datetime(row["start"], utc=True, errors="coerce")
                updated = pd.to_datetime(row["quote_time"], utc=True, errors="coerce")
                now = pd.Timestamp.now(tz="UTC")
                if start <= now or pd.isna(updated) or now - updated > pd.Timedelta(minutes=10):
                    st.error("Quote expired or game started. Refresh before logging.")
                else:
                    row.update(stake=stake, result="PENDING", logged_at=now.isoformat())
                    st.session_state.setdefault("sports_journal", []).append(row)
                    st.success("Paper single logged.")
        else:
            st.write("No fresh pregame selections available.")
    show_sports_journal(api_key)


def show_kalshi_quotes():
    with st.expander("Read-only Kalshi contract feed"):
        st.caption("Enter the exact contract ticker to fetch its quotes and rules. No account keys "
                   "are needed by this public endpoint. YES and NO do not always mean UP and DOWN.")
        ticker = st.text_input("Exact Kalshi ticker", key="quote_ticker").strip().upper()
        if st.button("Fetch Kalshi contract", disabled=not ticker):
            try:
                body, _, fetched = kalshi_feed(ticker)
                market = body["market"]
                st.write(market.get("title", ticker))
                st.caption(f"Fetched {fetched} | Status: {market.get('status')} | Closes: {market.get('close_time')}")
                prices = []
                for side in ("yes", "no"):
                    for action in ("bid", "ask"):
                        key = f"{side}_{action}"
                        value = market.get(key + "_dollars")
                        cents = float(value) * 100 if value is not None else market.get(key)
                        prices.append({"quote": key.upper(), "cents": cents})
                st.dataframe(pd.DataFrame(prices), use_container_width=True)
                st.write(market.get("rules_primary", "Rules unavailable."))
                st.write(market.get("rules_secondary", ""))
                st.caption(f"Result: {market.get('result') or 'Not settled'}. Quotes are snapshots, "
                           "not guaranteed fills. Fees and available size need checking on the exact contract.")
            except (FeedError, KeyError, ValueError) as exc:
                st.error(str(exc) if isinstance(exc, FeedError) else "Unexpected contract response.")


# Crypto discovery is separate from the BTC next-candle model. In particular,
# a quote midpoint must never be presented as an independently trained forecast.
KALSHI_ROOT = "https://external-api.kalshi.com/trade-api/v2"


@st.cache_data(ttl=600, show_spinner=False)
def crypto_series():
    body, _, fetched = get_json(KALSHI_ROOT + "/series?category=Crypto")
    series = body.get("series")
    if not isinstance(series, list):
        raise FeedError("Unexpected crypto series response")
    return series, fetched


def series_open_markets(series):
    rows, cursor, seen = [], "", set()
    for _ in range(20):
        params = dict(series_ticker=series["ticker"], status="open", limit=1000)
        if cursor:
            params["cursor"] = cursor
        try:
            body, _, fetched = get_json(KALSHI_ROOT + "/markets?" + urlencode(params))
        except FeedError as exc:
            return rows, str(exc)
        if not isinstance(body.get("markets"), list):
            return rows, "Unexpected market list response; scan incomplete"
        for market in body["markets"]:
            rows.append(dict(market, series_ticker=series["ticker"],
                             series_title=series.get("title", ""),
                             frequency=series.get("frequency", "unknown"),
                             quote_fetched_at=fetched))
        cursor = body.get("cursor", "")
        if not cursor:
            return rows, None
        if cursor in seen:
            return rows, "Repeated cursor: scan incomplete"
        seen.add(cursor)
    return rows, "Page limit reached: scan incomplete"


def scan_crypto(series):
    rows, issues = [], []
    # Serial scanning plus the shared request gate prevents request bursts.
    for index, item in enumerate(series):
        ticker = item["ticker"]
        try:
            found, warning = series_open_markets(item)
            rows.extend(found)
            if warning:
                issues.append(dict(series=ticker, issue=warning))
                if warning.startswith("Kalshi rate limit persists"):
                    # Stop a sustained limit instead of retrying every remaining series.
                    issues.extend(dict(series=s["ticker"], issue="Not scanned: rate-limit cooldown; retry later")
                                  for s in series[index + 1:])
                    break
        except Exception as exc:
            issues.append(dict(series=ticker, issue=str(exc) if isinstance(exc, FeedError)
                               else "Invalid response or unavailable feed"))
    unique = {r["ticker"]: r for r in rows if r.get("ticker")}
    return list(unique.values()), issues


def contract_price(market, name):
    value = market.get(name + "_dollars")
    legacy = value is None
    if legacy:
        value = market.get(name)
    try:
        value = float(value) / (100 if legacy else 1)
        return value if math.isfinite(value) and 0 <= value <= 1 else None
    except (ValueError, TypeError):
        return None


def crypto_call_meaning(market, side):
    title = market.get("title", "").lower()
    strike_type = market.get("strike_type", "")
    if "price up in next" in title and strike_type in ("greater", "greater_or_equal"):
        return "UP" if side == "YES" else "DOWN"
    if strike_type == "greater":
        return "Above strike" if side == "YES" else "At/below strike"
    if strike_type == "greater_or_equal":
        return "At/above strike" if side == "YES" else "Below strike"
    if strike_type == "between":
        return "Within range" if side == "YES" else "Outside range"
    return side + " — inspect settlement rules"


def crypto_candidate(market, threshold, max_spread, min_minutes, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    row = dict(ticker=market.get("ticker", ""), series=market.get("series_ticker", ""),
               title=market.get("title", ""), frequency=market.get("frequency", "unknown"),
               yes_meaning=market.get("yes_sub_title", ""),
               no_meaning=market.get("no_sub_title", ""),
               closes=market.get("close_time", ""), action="WAIT", side="",
               market_estimate=None, ask=None, spread=None, minutes_left=None,
               call_meaning="Unavailable", reason="No independent contract model validated",
               quote_fetched_at=market.get("quote_fetched_at", ""))
    bid, ask = contract_price(market, "yes_bid"), contract_price(market, "yes_ask")
    if bid is None or ask is None or bid > ask or bid <= 0 or ask >= 1:
        row["reason"] = "Missing, empty or crossed two-sided YES quote"
        return row
    p_yes = (bid + ask) / 2
    side = "YES" if p_yes >= .5 else "NO"
    side_bid, side_ask = contract_price(market, side.lower() + "_bid"), contract_price(market, side.lower() + "_ask")
    row.update(side=side, market_estimate=p_yes if side == "YES" else 1-p_yes,
               ask=side_ask, call_meaning=crypto_call_meaning(market, side))
    try:
        close = pd.to_datetime(market["close_time"], utc=True, errors="raise")
        fetched = pd.to_datetime(market["quote_fetched_at"], utc=True, errors="raise")
        if pd.isna(close) or pd.isna(fetched):
            raise ValueError()
        row["minutes_left"] = (close - now).total_seconds() / 60
    except (ValueError, TypeError, KeyError):
        row["reason"] = "Missing or invalid timing information"
        return row
    if side_bid is None or side_ask is None or not 0 < side_bid <= side_ask < 1:
        row["reason"] = "Selected side has no valid two-sided quote"
        return row
    row["spread"] = side_ask - side_bid
    if market.get("status") not in ("open", "active") or close <= now:
        row["reason"] = "Contract is not open"
    elif not 0 <= (now - fetched).total_seconds() <= 90:
        row["reason"] = "Quote snapshot is stale"
    elif row["minutes_left"] < min_minutes:
        row["reason"] = "Too close to expiration"
    elif row["spread"] > max_spread:
        row["reason"] = "Spread exceeds research limit"
    elif row["market_estimate"] < threshold:
        row["reason"] = "Market estimate is below watchlist threshold"
    else:
        row["action"] = "WATCH"
        row["reason"] = "Quote-based research candidate; model prediction unavailable"
    return row


def record_crypto_calls(rows, journal, fee, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    known = {r["ticker"] for r in journal}
    added = 0
    for r in rows:
        if r["action"] != "WATCH" or r["ticker"] in known:
            continue
        # Recheck timing at logging, not just at the earlier scan.
        if pd.Timestamp(r["closes"]) <= now or not 0 <= (now-pd.Timestamp(r["quote_fetched_at"])).total_seconds() <= 90:
            continue
        journal.append(dict(ticker=r["ticker"], series=r["series"],
                            frequency=r["frequency"], title=r["title"],
                            yes_meaning=r["yes_meaning"], no_meaning=r["no_meaning"],
                            recorded_at=now.isoformat(), closes=r["closes"],
                            side=r["side"], estimate=r["market_estimate"],
                            estimate_source="kalshi_quote_midpoint", ask=r["ask"],
                            fee=fee, result="", settled_at="", net_pnl=None))
        known.add(r["ticker"])
        added += 1
    return added


def settle_crypto_call(entry, market, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    result = market.get("result")
    if market.get("ticker") != entry["ticker"] or market.get("status") not in ("settled", "finalized"):
        return False
    if result not in ("yes", "no"):
        return False  # Keep voids and non-binary settlements unresolved.
    if pd.Timestamp(entry["recorded_at"]) >= pd.Timestamp(entry["closes"]):
        return False
    entry.update(result=result.upper(), settled_at=now.isoformat(),
                 net_pnl=float(result.upper() == entry["side"]) - entry["ask"] - entry["fee"])
    return True


def validate_crypto_journal(frame):
    required = {"ticker", "series", "frequency", "title", "yes_meaning", "no_meaning",
                "recorded_at", "closes", "side", "estimate", "estimate_source", "ask", "fee",
                "result", "settled_at", "net_pnl"}
    if not required.issubset(frame.columns) or len(frame) > 20000:
        raise ValueError("Use the exported crypto call journal (maximum 20,000 rows).")
    frame = frame.copy().fillna("")
    if frame.ticker.eq("").any() or frame.ticker.duplicated().any():
        raise ValueError("Contract tickers must be present and unique.")
    if not frame.side.isin(["YES", "NO"]).all() or not frame.estimate_source.eq("kalshi_quote_midpoint").all():
        raise ValueError("Unexpected forecast side or estimate source.")
    start = pd.to_datetime(frame.recorded_at, utc=True, errors="raise")
    close = pd.to_datetime(frame.closes, utc=True, errors="raise")
    if start.isna().any() or close.isna().any() or (start >= close).any():
        raise ValueError("Calls must have been recorded before their contract closed.")
    for name in ("estimate", "ask", "fee"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError("Amounts and estimates must be finite.")
    if not frame.estimate.between(.5, 1).all() or not frame.ask.between(.001, .999).all() or (frame.fee < 0).any():
        raise ValueError("Invalid estimates, entry prices or fees.")
    # Reconcile all imported settlements from Kalshi again; never trust CSV wins.
    frame["result"], frame["settled_at"], frame["net_pnl"] = "", "", None
    return frame[list(required)].to_dict("records")


def show_crypto_journal():
    journal = st.session_state.setdefault("crypto_calls", [])
    st.subheader("Exact-contract forward tracking")
    st.caption("One first call per contract, recorded before expiry. Results come from Kalshi settlement. "
               "This measures the quote-based watchlist, not independent bot accuracy. "
               "Entries assume one contract at the snapshot ask, without a guaranteed fill.")
    uploaded = st.file_uploader("Restore exported crypto call journal", type=["csv"], key="crypto_restore")
    if uploaded is not None and st.button("Restore crypto journal"):
        try:
            st.session_state.crypto_calls = validate_crypto_journal(pd.read_csv(uploaded))
            journal = st.session_state.crypto_calls
            st.success("Restored calls. Imported results were cleared for API verification.")
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    if st.button("Check pending settlements", disabled=not journal):
        pending = [r for r in journal if not r["result"] and pd.Timestamp(r["closes"]) < pd.Timestamp.now(tz="UTC")]
        errors, updated = 0, 0
        for entry in pending[:100]:
            try:
                body, _, _ = kalshi_feed(entry["ticker"])
                updated += int(settle_crypto_call(entry, body.get("market", {})))
            except (FeedError, ValueError, TypeError, KeyError):
                errors += 1
        st.info(f"Verified {updated} settlements. {errors} unavailable. "
                "Up to 100 pending contracts checked per click; older unavailable contracts remain pending.")
    if not journal:
        st.info("No calls recorded yet. Scan markets and log the watchlist to begin testing.")
        return
    frame = pd.DataFrame(journal)
    settled = frame[frame.result.isin(["YES", "NO"])].copy()
    if not settled.empty:
        settled["correct"] = settled.side == settled.result
        settled["brier"] = (settled.estimate - settled.correct.astype(float)) ** 2
        summary = []
        for keys, group in settled.groupby(["series", "frequency"]):
            n, wins = len(group), int(group.correct.sum())
            lo, hi = wilson(wins, n)
            summary.append(dict(series=keys[0], frequency=keys[1], settled=n,
                                accuracy_pct=round(100*wins/n, 1), lower_95_pct=round(100*lo, 1),
                                upper_95_pct=round(100*hi, 1), brier=round(group.brier.mean(), 4),
                                simulated_net=round(group.net_pnl.sum(), 2)))
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.caption("75–80% is a research target, not achieved status. Small samples, related strikes "
                   "and overlapping crypto contracts can overstate the evidence; Wilson intervals assume "
                   "independent calls. Evaluate each series and timeframe on new observations. "
                   "Accuracy alone does not imply profit; costs can exceed the win probability.")
    else:
        st.info("Accuracy unavailable until recorded contracts settle and are verified.")
    st.dataframe(frame.tail(200), use_container_width=True)
    st.download_button("Save crypto call journal", frame.to_csv(index=False),
                       "kalshi_crypto_calls.csv", "text/csv")
    st.warning("Journal is session-only. Download it before closing or restarting the app. "
               "Restored timestamps are user-supplied, so imported records are not an audited backtest.")


def show_crypto_scanner():
    st.header("All Kalshi crypto — discovery and forward research")
    st.info("Goal: test whether selective calls can reach 75–80% accuracy. No independent "
            "all-crypto settlement model has been validated yet. WATCH is a quote-based candidate; "
            "it is not a trade instruction or a claimed 75–80% bot prediction.")
    try:
        series, fetched = crypto_series()
    except FeedError as exc:
        st.error(str(exc))
        return
    st.caption(f"Discovered {len(series)} Crypto-category series at {fetched}. Discovery updates every 10 minutes.")
    frequencies = sorted({s.get("frequency", "unknown") for s in series})
    selected_freq = st.multiselect("Contract timeframes to scan", frequencies, default=frequencies)
    threshold = st.slider("Minimum MARKET estimate for watchlist (%)", 50, 95, 75, key="crypto_threshold") / 100
    max_spread = st.slider("Maximum selected-side spread (cents)", 1, 20, 5, key="crypto_spread") / 100
    min_minutes = st.number_input("Minimum minutes remaining", min_value=1, value=3, key="crypto_minutes")
    fee = st.number_input("Assumed total fee per simulated one-contract entry ($)",
                          min_value=0.0, value=0.02, step=0.01, key="crypto_fee")
    st.caption("Quotes can resemble probabilities but are not calibrated win rates. An 80% likely "
               "outcome bought at 85 cents loses 5 cents per contract on average before fees. "
               "Above/below a strike, within a range, and up/down are different settlement questions.")
    if st.button("Scan all selected crypto series", disabled=not selected_freq):
        with st.spinner("Scanning crypto contracts at a paced rate; retries may take extra time..."):
            markets, issues = scan_crypto([s for s in series if s.get("frequency", "unknown") in selected_freq])
        st.session_state.crypto_scan = dict(markets=markets, issues=issues,
                                             at=datetime.now(timezone.utc).isoformat())
    scan = st.session_state.get("crypto_scan")
    if scan:
        rows = [crypto_candidate(m, threshold, max_spread, int(min_minutes))
                for m in scan["markets"] if m["frequency"] in selected_freq]
        frame = pd.DataFrame(rows)
        st.caption(f"Last scan finished {scan['at']}. Refresh by scanning again; no background worker runs when the app is closed.")
        if scan["issues"]:
            st.warning("Coverage is incomplete. Failed series and truncated pages are listed below.")
            st.dataframe(pd.DataFrame(scan["issues"]), use_container_width=True)
        st.write(f"{len(rows)} open contracts returned; {sum(r['action'] == 'WATCH' for r in rows)} current watchlist candidates.")
        if not frame.empty:
            only = st.checkbox("Show only watchlist candidates", value=False)
            view = frame[frame.action == "WATCH"] if only else frame
            st.dataframe(view.sort_values(["action", "market_estimate"], ascending=[False, False]), use_container_width=True)
            st.download_button("Download all scan rows", frame.to_csv(index=False), "kalshi_crypto_scan.csv", "text/csv")
            if st.button("Record current watchlist for forward testing"):
                added = record_crypto_calls(rows, st.session_state.setdefault("crypto_calls", []), fee)
                st.success(f"Recorded {added} new first calls. Duplicate and stale contracts were skipped.")
            choice = st.selectbox("Inspect exact contract rules", frame.ticker.tolist())
            market = next(m for m in scan["markets"] if m["ticker"] == choice)
            with st.expander("Contract meanings and settlement rules"):
                st.write("YES: " + market.get("yes_sub_title", "Unavailable"))
                st.write("NO: " + market.get("no_sub_title", "Unavailable"))
                st.write(market.get("rules_primary", "Rules unavailable"))
                st.write(market.get("rules_secondary", ""))
                st.write({k: market.get(k) for k in ("strike_type", "floor_strike", "cap_strike", "close_time", "expiration_time")})
    with st.expander("Discovered series and settlement sources"):
        st.dataframe(pd.DataFrame([dict(series=s["ticker"], title=s.get("title"),
                                       frequency=s.get("frequency"),
                                       sources=json.dumps(s.get("settlement_sources", []))) for s in series]), use_container_width=True)
    show_crypto_journal()


def validate_forward_journal(frame):
    required = {"id", "opened_at", "interval", "contract", "side", "price_cents",
                "quantity", "entry_fee", "entry_cost", "status", "net_pnl",
                "bot_action", "p_up", "candle_end", "window_end"}
    if not required.issubset(frame.columns) or len(frame) > 10000:
        raise ValueError("Use an exported BTC forward journal with at most 10,000 rows.")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Journal IDs must be present and unique.")
    pd.to_datetime(frame.opened_at, utc=True, errors="raise")
    if not frame.interval.isin(["15m", "1h"]).all() or not frame.side.isin(["YES", "NO"]).all():
        raise ValueError("Invalid timeframe or contract side.")
    if not frame.status.isin(["OPEN", "WIN", "LOSS", "SELL"]).all():
        raise ValueError("Invalid trade status.")
    for field in ("price_cents", "quantity", "entry_fee", "entry_cost"):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(frame[field]).all():
            raise ValueError("Trade amounts must be finite.")
    if not frame.price_cents.between(1, 99).all() or not frame.quantity.between(1, 10000).all():
        raise ValueError("Invalid entry price or quantity.")
    if (frame.quantity % 1 != 0).any() or (frame.entry_fee < 0).any():
        raise ValueError("Invalid quantity or fee.")
    expected = frame.quantity * frame.price_cents / 100 + frame.entry_fee
    if not np.allclose(expected, frame.entry_cost):
        raise ValueError("Entry costs do not match prices and fees.")
    settled = frame.status != "OPEN"
    frame["net_pnl"] = pd.to_numeric(frame.net_pnl, errors="raise")
    if not np.isfinite(frame.loc[settled, "net_pnl"]).all():
        raise ValueError("Settled entries need finite P/L.")
    return frame.to_dict("records")


def show_forward_journal(snapshot, max_daily):
    st.subheader("Your manual simulated trades")
    st.caption("Open a simulated entry now; record its observed result later using the exact contract rules. "
               "The bot never sends orders. Download your journal before closing the app and import it next time.")
    journal = st.session_state.setdefault("btc_forward_journal", [])
    upload = st.file_uploader("Import BTC forward journal CSV", type="csv", key="btc_forward_import")
    if st.button("Load saved BTC journal", disabled=upload is None):
        try:
            rows = validate_forward_journal(pd.read_csv(upload))
            ids = {r["id"] for r in journal}
            journal.extend(r for r in rows if r["id"] not in ids)
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Could not import journal: {exc}")
    with st.form("forward_entry"):
        ticker = st.text_input("Exact contract ticker and expiration", key="forward_ticker")
        side = st.selectbox("Contract side to simulate", ["YES", "NO"], key="forward_side")
        st.caption("Check the contract wording: YES does not necessarily mean UP. "
                   "The BTC model percentage is not used to calculate a contract edge.")
        price = st.number_input("Observed ask (cents)", 1, 99, 50, key="forward_price")
        qty = st.number_input("Simulated contracts", 1, 10000, 1, key="forward_qty")
        fee = st.number_input("Observed total entry fee ($)", min_value=0.0, value=0.0, step=0.01, key="forward_fee")
        cap = st.number_input("Maximum cost for this simulated entry ($)", min_value=0.0, value=10.0, key="forward_cap")
        submit = st.form_submit_button("Open simulated trade")
    if submit:
        now = pd.Timestamp.now(tz="UTC")
        today = now.tz_convert("America/New_York").date()
        used = sum(pd.Timestamp(r["opened_at"]).tz_convert("America/New_York").date() == today
                   and r["interval"] == snapshot["interval"] for r in journal)
        cost = qty * price / 100 + fee
        if not ticker.strip() or cost > cap or used >= max_daily:
            st.error("Enter a ticker, keep cost within your entry limit, and stay within the daily entry limit.")
        else:
            journal.append(dict(id=uuid.uuid4().hex, opened_at=now.isoformat(),
                interval=snapshot["interval"], contract=ticker.strip(), side=side,
                price_cents=price, quantity=qty, entry_fee=fee, entry_cost=cost,
                status="OPEN", net_pnl=None, bot_action=snapshot["action"],
                p_up=snapshot["p_up"], candle_end=snapshot["candle_end"], window_end=snapshot["window_end"]))
            st.success("Simulated entry recorded. No order was placed.")
    opened = [r for r in journal if r["status"] == "OPEN"]
    if opened:
        with st.form("forward_settle"):
            ident = st.selectbox("Open trade to update", [r["id"] for r in opened],
                format_func=lambda v: next(f"{r['contract']} | {r['side']} | {r['opened_at']}" for r in opened if r["id"] == v))
            result = st.selectbox("Observed outcome for your purchased side", ["WIN", "LOSS", "SELL"])
            exit_price = st.number_input("Observed exit bid (cents, SELL only)", 0, 100, 50)
            exit_fee = st.number_input("Observed total exit fee ($, SELL only)", min_value=0.0, value=0.0, step=0.01)
            observed = st.checkbox("I checked settlement or recorded the available exit quote")
            settle = st.form_submit_button("Record simulated outcome")
        if settle:
            if not observed:
                st.error("Check the observed outcome before recording it.")
            else:
                row = next(r for r in journal if r["id"] == ident)
                _, pnl = paper_pnl(row["price_cents"], row["quantity"], row["entry_fee"], result, exit_price, exit_fee)
                row.update(status=result, net_pnl=pnl, closed_at=datetime.now(timezone.utc).isoformat(),
                           exit_cents=exit_price if result == "SELL" else None,
                           exit_fee=exit_fee if result == "SELL" else 0)
                st.success("Observed simulated result recorded.")
    if journal:
        frame = pd.DataFrame(journal)
        settled = frame.loc[frame.status != "OPEN"]
        resolved = settled.loc[settled.status.isin(["WIN", "LOSS"])]
        cols = st.columns(3)
        cols[0].metric("Settled simulated trades", len(settled))
        cols[1].metric("Simulated net P/L", f"${settled.net_pnl.sum():+.2f}")
        cols[2].metric("Settlement win rate", f"{100 * (resolved.status == 'WIN').mean():.1f}%" if len(resolved) else "No settlements")
        st.caption("Win rate excludes early sales. This measures your manual contract selections, "
                   "not the BTC model's candle accuracy. Quotes are snapshots; fills are not guaranteed.")
        st.dataframe(frame, use_container_width=True)
        st.download_button("Download BTC forward journal CSV", frame.to_csv(index=False), "btc_forward_journal.csv", "text/csv")


# Forward observations compare the Yahoo direction proxy with official Kalshi
# outcomes. These observations do not turn proxy scores into contract probabilities.
TRACK_FIELDS = ['ticker', 'recorded_at', 'window_start', 'window_end', 'target',
                'p_up_proxy', 'proxy_side', 'bot_action', 'skip_reason', 'yes_ask',
                'no_ask', 'rules_primary', 'result', 'verified_at', 'origin']


# A display-only abstention band. It is not a validated trading threshold and
# does not change probabilities, original observations, or the evidence gate.
def tracker_direction(p, minimum=0.55):
    if p is None or not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
        return 'UNAVAILABLE'
    if float(p) >= minimum:
        return 'UP'
    if 1 - float(p) >= minimum:
        return 'DOWN'
    return 'NEUTRAL'


def tracker_match(market, snapshot, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    start, end = pd.Timestamp(snapshot['candle_end']), pd.Timestamp(snapshot['window_end'])
    close = pd.to_datetime(market.get('close_time'), utc=True, errors='coerce')
    rules = market.get('rules_primary') or ''
    # Only the standard BTC 15-minute series and the exact forecast end qualify.
    if snapshot['interval'] != '15m' or not str(market.get('ticker', '')).startswith('KXBTC15M-'):
        return False
    if pd.isna(close) or close != end or not start <= now < end:
        return False
    if not 0 <= (now - start).total_seconds() <= 120:
        return False  # Do not backfill or silently record a late forecast.
    if market.get('status') not in ('active', 'open') or market.get('strike_type') != 'greater_or_equal':
        return False
    text = rules.lower()
    if 'brti' not in text or 'sixty seconds' not in text or 'at least' not in text:
        return False
    # Check both comparison times in the actual rules, not just the ticker.
    for stamp in (start, end):
        eastern = stamp.tz_convert('America/New_York')
        label = eastern.strftime('%I:%M %p').lstrip('0').lower()
        if 'before ' + label not in text:
            return False
    return True


def tracker_record(journal, market, snapshot, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if not tracker_match(market, snapshot, now):
        return False
    if any(r['ticker'] == market['ticker'] for r in journal):
        return False  # Retain the first observation, including SKIP.
    p = snapshot.get('p_up')
    if p is None or not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
        return False
    journal.append(dict(ticker=market['ticker'], recorded_at=now.isoformat(),
        window_start=snapshot['candle_end'], window_end=snapshot['window_end'],
        target=market.get('floor_strike'), p_up_proxy=float(p),
        proxy_side='YES' if p >= .5 else 'NO', bot_action=snapshot['action'],
        skip_reason=snapshot['reason'], yes_ask=contract_price(market, 'yes_ask'),
        no_ask=contract_price(market, 'no_ask'), rules_primary=market.get('rules_primary', ''),
        result='', verified_at='', origin='live_session'))
    return True


def tracker_settle(row, market, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now < pd.Timestamp(row['window_end']) or market.get('ticker') != row['ticker']:
        return False
    if market.get('status') not in ('settled', 'finalized') or market.get('result') not in ('yes', 'no'):
        return False
    row.update(result=market['result'].upper(), verified_at=now.isoformat())
    return True


def tracker_restore(frame):
    if len(frame) > 20000 or not set(TRACK_FIELDS).issubset(frame.columns):
        raise ValueError('Use a downloaded BTC tracking CSV, with at most 20,000 rows.')
    frame = frame[TRACK_FIELDS].copy().fillna('')
    if frame.ticker.duplicated().any() or not frame.ticker.str.startswith('KXBTC15M-').all():
        raise ValueError('Tracking requires unique BTC 15-minute contract tickers.')
    times = {k: pd.to_datetime(frame[k], utc=True, errors='raise')
             for k in ('recorded_at', 'window_start', 'window_end')}
    if any(v.isna().any() for v in times.values()):
        raise ValueError('Missing forecast timestamps.')
    age = (times['recorded_at'] - times['window_start']).dt.total_seconds()
    if not age.between(0, 120).all() or not ((times['window_end'] - times['window_start']) == pd.Timedelta(minutes=15)).all():
        raise ValueError('Forecasts must have been logged within two minutes of a 15-minute window start.')
    frame['p_up_proxy'] = pd.to_numeric(frame.p_up_proxy, errors='raise')
    if not frame.p_up_proxy.between(0, 1).all():
        raise ValueError('Invalid model estimates.')
    expected = np.where(frame.p_up_proxy >= .5, 'YES', 'NO')
    if not (frame.proxy_side == expected).all() or not frame.bot_action.isin(['SKIP', 'UP', 'DOWN']).all():
        raise ValueError('Invalid model directions.')
    # CSV timestamps and predictions are user-supplied. Verify results anew.
    frame['origin'] = 'imported_unverified_forecast'
    frame['result'], frame['verified_at'] = '', ''
    return frame.to_dict('records')


@st.cache_data(ttl=30, show_spinner=False)
def tracker_candles():
    # Fetch directly: nesting two candle caches can retain the prior window.
    raw = yf.download("BTC-USD", period="60d", interval="15m",
                      auto_adjust=True, progress=False)
    fetched_at = datetime.now(timezone.utc).isoformat()
    if raw is None or raw.empty:
        raise FeedError('BTC candle feed unavailable. Tracking will retry.')
    data = indicators(closed_candles(raw, '15m'))
    if data.empty:
        raise FeedError('Not enough completed candles. Tracking will retry.')
    return data, fetched_at


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def tracker_fit(data):
    # The DataFrame is the cache key: a new candle causes a new fit.
    return model_research(data, '15m', 4)


def tracker_model():
    data, fetched_at = tracker_candles()
    end = data.index[-1] + pd.Timedelta(minutes=15)
    st.session_state.tracking_feed = dict(fetched_at=fetched_at, candle_end=end.isoformat())
    return tracker_fit(data), end


def tracker_tick(journal):
    messages = []
    # Rotate pending checks so unavailable old contracts cannot starve newer ones.
    now = pd.Timestamp.now(tz='UTC')
    pending = [r for r in journal if not r['result'] and pd.Timestamp(r['window_end']) <= now]
    offset = st.session_state.get('tracking_check_offset', 0)
    if pending:
        chosen = (pending + pending)[offset % len(pending):offset % len(pending) + min(3, len(pending))]
        st.session_state.tracking_check_offset = offset + len(chosen)
        for row in chosen:
            try:
                body, _, _ = kalshi_feed(row['ticker'])
                tracker_settle(row, body.get('market', {}))
            except RateLimitError as exc:
                messages.append(str(exc))
                return messages
            except FeedError:
                messages.append('A settlement is unavailable; it remains pending for a later check.')
    # Outside the first two minutes, only check settlements. Fitting the model
    # every 30 seconds cannot produce an eligible new observation then.
    window_start = pd.Timestamp.now(tz='UTC').floor('15min')
    if (pd.Timestamp.now(tz='UTC') - window_start).total_seconds() > 120:
        return messages
    try:
        report, candle_end = tracker_model()
        snapshot = live_snapshot(report, candle_end, '15m', 4, [])
        snapshot['training_up_rate'] = report.get('base_probability') if report else None
        st.session_state.tracking_snapshot = snapshot
        now = pd.Timestamp.now(tz='UTC')
        if not 0 <= (now - pd.Timestamp(snapshot['candle_end'])).total_seconds() <= 120:
            return messages + ['The available candle does not match an early current window. Tracking will retry; expired forecasts are not recorded.']
        if any(r['window_end'] == snapshot['window_end'] for r in journal):
            return messages + ['This window is already saved. Waiting for the next window.']
        body, _, _ = get_json(KALSHI_ROOT + '/markets?' + urlencode(
            {'series_ticker': 'KXBTC15M', 'status': 'open', 'limit': 100}))
        matches = []
        for item in body.get('markets', []):
            close = pd.to_datetime(item.get('close_time'), utc=True, errors='coerce')
            if pd.notna(close) and close == pd.Timestamp(snapshot['window_end']):
                detail, _, _ = kalshi_feed(item['ticker'])
                market = detail.get('market', {})
                if tracker_match(market, snapshot):
                    matches.append(market)
        if len(matches) == 1 and tracker_record(journal, matches[0], snapshot):
            messages.append('Saved this window, including the bot’s SKIP status when applicable.')
        else:
            messages.append('No single verified matching contract yet. Tracking will retry; no forecast was saved.')
    except (FeedError, ValueError, TypeError, KeyError) as exc:
        messages.append(str(exc) if isinstance(exc, FeedError) else 'Data not ready. Tracking will retry.')
    return messages


def show_btc_tracking():
    st.header('BTC contract tracking')
    st.write('Turn on tracking and leave this page open. The app saves one forecast per matching '
             '15-minute contract, including SKIP, and checks official results after closing.')
    st.caption('No bets or orders. Tracking pauses if the session disconnects or your phone suspends the page. '
               'Missed windows are not filled in afterward. This compares a Yahoo direction proxy with Kalshi outcomes.')
    auto_supported = hasattr(st, 'fragment')
    st.checkbox('Start tracking', key='tracking_enabled', disabled=not auto_supported)
    show_rows = st.checkbox('Show saved windows', value=False, key='tracking_show_rows')
    st.caption('Quiet view keeps the saved-window table hidden. Scores between 45% and 55% '
               'display as NEUTRAL; this display filter has not been validated as a strategy.')
    if not auto_supported:
        st.info('Automatic refresh needs Streamlit 1.37 or newer. You can use Check now meanwhile.')
    with st.expander('Restore a saved tracking file'):
        upload = st.file_uploader('BTC tracking CSV', type=['csv'], key='tracking_upload')
        if st.button('Restore tracking', disabled=upload is None):
            try:
                rows = tracker_restore(pd.read_csv(upload))
                journal = st.session_state.setdefault('btc_contract_tracking', [])
                known = {r['ticker'] for r in journal}
                journal.extend(r for r in rows if r['ticker'] not in known)
                st.success('Restored. Outcomes will be checked again; imported forecasts are labeled separately.')
            except (ValueError, TypeError, KeyError):
                st.error('Could not restore this file. Use the CSV downloaded from this tracker.')
    enabled = st.session_state.get('tracking_enabled', False)
    decorator = st.fragment(run_every=30 if enabled else None) if auto_supported else lambda f: f

    @decorator
    def panel():
        journal = st.session_state.setdefault('btc_contract_tracking', [])
        check = st.button('Check now', key='tracking_check')
        if check:
            tracker_candles.clear()
            kalshi_feed.clear()
        if enabled or check:
            with st.spinner('Checking forecast and contract results...') if check else nullcontext():
                messages = tracker_tick(journal)
                if messages:
                    st.session_state.tracking_status = messages[-1]
            st.session_state.tracking_last_check = datetime.now(timezone.utc).isoformat()
        if check and st.session_state.get('tracking_status'):
            st.info(st.session_state.tracking_status)
        st.caption('Tracking is ' + ('ON — checks about every 30 seconds.' if enabled else 'OFF.'))
        if st.session_state.get('tracking_last_check'):
            stamp = pd.Timestamp(st.session_state.tracking_last_check).tz_convert('America/New_York')
            st.caption(f'Last check: {stamp:%I:%M:%S %p %Z}')
        feed = st.session_state.get('tracking_feed')
        if feed:
            candle = pd.Timestamp(feed['candle_end']).tz_convert('America/New_York')
            fetched = pd.Timestamp(feed['fetched_at']).tz_convert('America/New_York')
            st.caption(f'Latest completed candle ended: {candle:%I:%M %p %Z}. '
                       f'Feed fetched: {fetched:%I:%M:%S %p %Z}.')
        snap = st.session_state.get('tracking_snapshot')
        if snap:
            st.write('Trade action: ' + snap['action'])
            p = snap.get('p_up')
            direction = tracker_direction(p)
            expired = pd.Timestamp.now(tz='UTC') >= pd.Timestamp(snap['window_end'])
            st.write(('Last observed research direction: ' if expired else 'Research direction: ') + direction)
            if p is not None:
                st.caption(f'Raw next-candle UP estimate: {100*p:.2f}%. '
                           'This is a Yahoo candle proxy, not a calibrated Kalshi probability.')
            if direction == 'NEUTRAL':
                st.info('No clear directional estimate. A tiny lean above or below 50% is treated as neutral.')
            st.caption(snap['reason'])
            baseline = snap.get('training_up_rate')
            if baseline is not None:
                st.caption(f'Training candles ending UP: {100*baseline:.1f}%. '
                           'A slight UP lean can reflect the training distribution; it does not prove an edge.')
            start = pd.Timestamp(snap['candle_end']).tz_convert('America/New_York')
            end = pd.Timestamp(snap['window_end']).tz_convert('America/New_York')
            st.caption(f'Forecast window: {start:%I:%M %p}–{end:%I:%M %p %Z}.')
        frame = pd.DataFrame(journal, columns=TRACK_FIELDS)
        resolved = frame[frame.result.isin(['YES', 'NO'])]
        native = resolved[resolved.origin == 'live_session']
        active = native[native.bot_action.isin(['UP', 'DOWN'])]
        cols = st.columns(3)
        cols[0].metric('Windows saved', len(frame))
        cols[1].metric('Results verified', len(resolved))
        cols[2].metric('Qualifying calls verified', len(active))
        if len(native):
            wins = int((native.proxy_side == native.result).sum())
            st.write(f'All-window proxy agreement: {wins}/{len(native)} ({100*wins/len(native):.1f}%). Includes SKIP windows.')
            directions = native.p_up_proxy.map(tracker_direction)
            neutral_n = int((directions == 'NEUTRAL').sum())
            st.caption(f'Neutral observations under the current display filter: {neutral_n}/{len(native)}. '
                       'The all-window comparison above retains the original 50% split for audit purposes.')
            yes_wins = int((native.result == 'YES').sum())
            st.caption(f'Always-YES comparison on these same windows: {yes_wins}/{len(native)} '
                       f'({100*yes_wins/len(native):.1f}%).')
            if len(active):
                wins = int((active.proxy_side == active.result).sum())
                low, high = wilson(wins, len(active))
                st.write(f'Qualifying-call agreement: {wins}/{len(active)} ({100*wins/len(active):.1f}%). '
                         f'95% interval: {100*low:.1f}–{100*high:.1f}%.')
            else:
                st.info('No qualifying calls have settled. Saved SKIP observations are research data, not trade signals.')
            st.caption('These are forward proxy comparisons, not calibrated contract probabilities or profitability. '
                       'Imported forecasts are excluded from these metrics; nearby windows may be correlated.')
        if len(frame):
            if show_rows:
                display = frame[['ticker', 'recorded_at', 'bot_action', 'p_up_proxy', 'result', 'origin']].copy()
                display['research_direction'] = display.p_up_proxy.map(tracker_direction)
                st.dataframe(display.tail(100), use_container_width=True)
            st.download_button('Save tracking file', frame.to_csv(index=False), 'btc_contract_tracking.csv', 'text/csv',
                               key='tracking_download')
        st.warning('Save your tracking file before closing. Records are held in this session and can be lost on restart. '
                   'Restore the CSV next time to continue checking pending outcomes.')
    panel()


def tt_ratings(history, cutoff):
    """Pregame Elo; use only completed matches before the selected start."""
    required = {"completed_at", "player_a", "player_b", "winner"}
    if not required.issubset(history.columns):
        raise ValueError("CSV needs: completed_at, player_a, player_b, winner")
    h = history.copy()
    h["completed_at"] = pd.to_datetime(h.completed_at, utc=True, errors="raise")
    if h.completed_at.isna().any():
        raise ValueError("Every match needs a completion timestamp.")
    for col in ["player_a", "player_b", "winner"]:
        if h[col].isna().any():
            raise ValueError("Player names and winners cannot be blank.")
        h[col] = h[col].astype(str).str.strip()
    invalid = (h.player_a == h.player_b) | (h.player_a == "") | (h.player_b == "") | ~((h.winner == h.player_a) | (h.winner == h.player_b))
    if invalid.any():
        raise ValueError("Each row needs two distinct players; winner must exactly match one name.")
    if h.duplicated(["completed_at", "player_a", "player_b"]).any():
        raise ValueError("Duplicate match rows found. Remove duplicates first.")
    h = h[h.completed_at < cutoff].sort_values("completed_at", kind="stable")
    ratings, counts = {}, {}
    for row in h.itertuples():
        a, b = row.player_a, row.player_b
        ra, rb = ratings.get(a, 1500.0), ratings.get(b, 1500.0)
        expected = 1 / (1 + 10 ** ((rb - ra) / 400))
        change = 24 * (float(row.winner == a) - expected)
        ratings[a], ratings[b] = ra + change, rb - change
        counts[a], counts[b] = counts.get(a, 0) + 1, counts.get(b, 0) + 1
    return ratings, counts, h


def tt_decimal(american):
    if abs(american) < 100:
        raise ValueError("American odds must be at least +100 or at most -100.")
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def tt_api(path, params=None):
    """Public read-only Kalshi data. Never sends account credentials."""
    url = 'https://external-api.kalshi.com/trade-api/v2/' + path
    if params:
        url += '?' + urlencode(params)
    try:
        req = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'TradingSportsLab/12'})
        with urlopen(req, timeout=15) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError('Unexpected market data format.')
        return payload
    except HTTPError as exc:
        raise ValueError(f'Kalshi data request failed (HTTP {exc.code}). Try again later.') from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise ValueError('Kalshi data is unavailable. No fresh picks can be generated.') from None


@st.cache_data(ttl=900, show_spinner=False)
def tt_catalog():
    import re
    rows = tt_api('series', {'category': 'Sports'}).get('series')
    if not isinstance(rows, list):
        raise ValueError('Kalshi did not return its sports catalog.')
    found = []
    for row in rows:
        text = json.dumps(row).lower()
        title = row.get('title', '').lower()
        if any(k in text for k in ['table tennis', 'tabletennis', 'tt elite', 'tt star', 'liga pro']):
            if 'match' in title and not any(k in title for k in ['game', 'champion', 'tournament winner']):
                found.append(row)
    return sorted(found, key=lambda x: x.get('title', ''))


@st.cache_data(ttl=60, show_spinner=False)
def tt_open(series):
    return tt_pages(series, 'open', 4)


@st.cache_data(ttl=1800, show_spinner=False)
def tt_history_auto(series):
    return tt_pages(series, 'settled', 6)


def tt_pages(series, status, max_pages):
    rows, cursor, seen = [], '', set()
    for _ in range(max_pages):
        args = {'series_ticker': series, 'status': status, 'limit': 1000}
        if cursor:
            args['cursor'] = cursor
        payload = tt_api('markets', args)
        page = payload.get('markets')
        if not isinstance(page, list):
            raise ValueError('Kalshi returned an invalid markets list.')
        rows.extend(page)
        cursor = payload.get('cursor') or ''
        if not cursor:
            return rows, False, pd.Timestamp.now(tz='UTC').isoformat()
        if cursor in seen:
            raise ValueError('Kalshi pagination repeated; scan stopped.')
        seen.add(cursor)
    return rows, True, pd.Timestamp.now(tz='UTC').isoformat()


def tt_pairs(markets):
    """Group the two separate YES-player contracts; NO subtitle is NOT the opponent."""
    groups = {}
    for m in markets:
        if m.get('market_type') != 'binary':
            continue
        ident = (m.get('custom_strike') or {}).get('table_tennis_competitor')
        name = m.get('yes_sub_title')
        rules = m.get('rules_primary', '').lower()
        if not ident or not name or 'wins the ' not in rules or 'match' not in rules:
            continue
        event = m.get('event_ticker')
        if event:
            groups.setdefault(event, {})[m.get('ticker')] = m
    pairs = []
    for event, group in groups.items():
        items = sorted(group.values(), key=lambda m: m['ticker'])
        if len(items) != 2:
            continue
        ids = [(m['custom_strike']['table_tennis_competitor']) for m in items]
        if ids[0] != ids[1]:
            pairs.append((event, items))
    return pairs


def tt_start(market):
    import re
    # occurrence_datetime is often an estimated END time. Parse only explicit scheduled start.
    match = re.search(r'originally scheduled for ([A-Za-z]+ \d{1,2}, \d{4}) at (\d{1,2}:\d{2} [AP]M) (EDT|EST)', market.get('rules_primary', ''))
    if not match:
        return None
    stamp = pd.Timestamp(datetime.strptime(match[1] + ' ' + match[2], '%b %d, %Y %I:%M %p'))
    offset = '-04:00' if match[3] == 'EDT' else '-05:00'
    return pd.Timestamp(stamp.isoformat() + offset).tz_convert('UTC')


def tt_auto_history(markets, now):
    records = []
    for event, pair in tt_pairs(markets):
        results = [m.get('result') for m in pair]
        if results.count('yes') != 1 or results.count('no') != 1:
            continue
        try:
            values = [float(m['settlement_value_dollars']) for m in pair]
            if sorted(values) != [0.0, 1.0]:
                continue  # excludes void and partial settlements
            stamps = [pd.Timestamp(m['settlement_ts']) for m in pair]
            if any(s.tzinfo is None for s in stamps):
                continue
            settled = max(stamps).tz_convert('UTC')
            if settled >= now:
                continue
            ids = [m['custom_strike']['table_tennis_competitor'] for m in pair]
            winner = ids[results.index('yes')]
            records.append({'completed_at': settled, 'player_a': ids[0], 'player_b': ids[1], 'winner': winner, 'event': event})
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(records, columns=['completed_at', 'player_a', 'player_b', 'winner', 'event']).drop_duplicates('event')


def tt_auto_rows(open_markets, historical_markets, now, fetched_at):
    h = tt_auto_history(historical_markets, now)
    ratings, counts, _ = tt_ratings(h, now)
    rows = []
    for event, pair in tt_pairs(open_markets):
        starts = [tt_start(m) for m in pair]
        if None in starts or starts[0] != starts[1] or starts[0] <= now + pd.Timedelta(minutes=2):
            continue
        ids = [m['custom_strike']['table_tennis_competitor'] for m in pair]
        n = [counts.get(i, 0) for i in ids]
        model_ok = min(n) >= 20
        p = 1 / (1 + 10 ** ((ratings.get(ids[1], 1500) - ratings.get(ids[0], 1500)) / 400)) if model_ok else None
        for index, m in enumerate(pair):
            try:
                ask = float(m['yes_ask_dollars'])
                size = float(m['yes_ask_size_fp'])
                bid = float(m['yes_bid_dollars'])
            except (KeyError, TypeError, ValueError):
                ask, size, bid = float('nan'), 0, float('nan')
            probability = (p if index == 0 else 1 - p) if p is not None else None
            quote_ok = all(math.isfinite(v) for v in [ask, bid, size]) and 0 < ask < 1 and 0 <= bid <= ask and size >= 1 and m.get('status') == 'active'
            reason = 'WAIT — insufficient history' if not model_ok else ('WAIT — no full-contract quote' if not quote_ok else 'Research estimate — fees not included')
            rows.append({'event': event, 'ticker': m['ticker'], 'player': m['yes_sub_title'], 'opponent': pair[1-index]['yes_sub_title'],
                         'start': starts[0].isoformat(), 'probability': probability, 'ask': ask if math.isfinite(ask) else None,
                         'margin_before_fees': probability - ask if probability is not None and quote_ok else None,
                         'player_matches': n[index], 'opponent_matches': n[1-index], 'quote_size': size,
                         'status': reason, 'fetched_at': fetched_at})
    return pd.DataFrame(rows), h


def show_table_tennis_auto():
    import altair as alt
    st.subheader('Automatic match scanner')
    st.caption('Public Kalshi schedules, prices and settled results · no API key required')
    st.info('Elo estimates are experimental. The bot skips started matches, incomplete results, thin quotes and players with fewer than 20 available historical matches. Fees are excluded; these are paper research picks.')
    st.caption('DraftKings: these player estimates can help research the same matchup, but DraftKings listings and odds are not connected or verified here.')
    try:
        catalog = tt_catalog()
        if not catalog:
            st.warning('No supported table tennis match series found in the current Kalshi catalog.')
            return
        mapping = {s['ticker']: s for s in catalog}
        default = next((i for i, s in enumerate(catalog) if s['ticker'] == 'KXLIGAPROMATCH'), 0)
        league = st.selectbox('League', list(mapping), index=default, format_func=lambda t: mapping[t]['title'], key='tt_auto_league')
        if st.button('Refresh matches and history', key='tt_auto_refresh'):
            tt_open.clear()
            tt_history_auto.clear()
        refresh_on = st.checkbox('Refresh prices every minute while this page is open', value=True, key='tt_auto_poll')
        @st.fragment(run_every=60 if refresh_on else None)
        def board():
            try:
                markets, truncated, captured = tt_open(league)
                now = pd.Timestamp.now(tz='UTC')
                if now - pd.Timestamp(captured) > pd.Timedelta(seconds=90):
                    st.warning('WAIT — quote snapshot is stale. Refresh to continue.')
                    return
                if truncated:
                    st.warning('Open-market scan reached its page limit; some matches may be missing.')
                st.caption(f'Prices fetched {pd.Timestamp(captured).tz_convert("America/New_York"):%b %d, %I:%M:%S %p %Z}. Refresh runs while this page is active; this is not background hosting.')
                if not markets:
                    st.info('No open matches returned for this league. Try another league.')
                    return
                with st.spinner('Loading settled match history...'):
                    historical, history_limited, history_time = tt_history_auto(league)
                now = pd.Timestamp.now(tz='UTC')
                if now - pd.Timestamp(captured) > pd.Timedelta(seconds=90):
                    st.warning('WAIT — quotes expired while history loaded. Refresh prices to continue.')
                    return
                frame, history = tt_auto_rows(markets, historical, now, captured)
                st.caption(f'History: {len(history):,} complete matches available in the returned Kalshi sample. Older archived matches are not included. Elo starts at 1500 at the beginning of this sample.')
                if history_limited:
                    st.warning('History scan reached its page limit. Estimates use a partial history sample.')
                if frame.empty:
                    st.info('WAIT — no supported upcoming paired match markets with a verified start more than two minutes away. Started matches and unrecognized rules are excluded.')
                    return
                candidates = frame.dropna(subset=['probability']).sort_values(['event', 'probability'], ascending=[True, False]).drop_duplicates('event')
                st.subheader('Match-winner research picks')
                if candidates.empty:
                    st.warning('WAIT — the available history does not yet support a winner estimate for these players.')
                else:
                    display = candidates[['player','opponent','start','probability','player_matches','opponent_matches','status']].copy()
                    display['start'] = pd.to_datetime(display.start, utc=True).dt.tz_convert('America/New_York').dt.strftime('%b %d %I:%M %p %Z')
                    display['probability'] = display.probability.map(lambda p: f'{p:.1%}')
                    display.columns = ['Research winner','Opponent','Start (New York)','Elo estimate','Player history','Opponent history','Price check']
                    st.dataframe(display, use_container_width=True, hide_index=True)
                label_map = {event: f"{group.iloc[0].player} vs {group.iloc[0].opponent}" for event, group in frame.groupby('event')}
                event = st.selectbox('Inspect match', list(label_map), format_func=lambda t: label_map[t], key='tt_inspect_' + league)
                chosen = frame[frame.event == event]
                ready = chosen.dropna(subset=['probability'])
                if not ready.empty:
                    bars = ready[['player','probability']].rename(columns={'player':'Player','probability':'Win estimate'})
                    base = alt.Chart(bars).encode(y=alt.Y('Player:N', title=None), x=alt.X('Win estimate:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%'), title='Experimental Elo win estimate'))
                    st.altair_chart((base.mark_bar(color='#2563eb') + base.mark_text(align='right', dx=-5, color='white').encode(text=alt.Text('Win estimate:Q', format='.1%'))).properties(height=160), use_container_width=True)
                prices = chosen[['player','ask','quote_size','margin_before_fees','status']].copy()
                prices['ask'] = prices.ask.map(lambda x: f'{100*x:.1f}¢' if pd.notna(x) else 'Unavailable')
                prices['margin_before_fees'] = prices.margin_before_fees.map(lambda x: f'{100*x:+.1f} percentage points' if pd.notna(x) else 'Unavailable')
                st.dataframe(prices.rename(columns={'player':'YES player','ask':'Kalshi ask','quote_size':'Ask size','margin_before_fees':'Model margin before fees','status':'Status'}), hide_index=True, use_container_width=True)
                with st.expander('Contract rules'):
                    for m in markets:
                        if m.get('event_ticker') == event:
                            st.write(m.get('title'))
                            st.code(m['ticker'], language=None)
                            st.write(m.get('rules_primary', ''))
                            st.write(m.get('rules_secondary', ''))
                export = frame.copy()
                export['result'] = 'PENDING'
                export['model'] = 'Kalshi-history Elo K=24, experimental'
                st.download_button('Save paper research snapshot', export.to_csv(index=False), 'table_tennis_auto_picks.csv', 'text/csv', key='tt_auto_export_' + league)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                st.warning(f'Automatic scanner paused: {exc}')
        board()
    except (ValueError, KeyError, TypeError) as exc:
        st.warning(f'Automatic scanner unavailable: {exc}')


@st.cache_data(ttl=900, show_spinner=False)
def tennis_catalog():
    rows = tt_api('series', {'category': 'Sports'}).get('series')
    if not isinstance(rows, list):
        raise ValueError('Kalshi did not return its sports catalog.')
    found = []
    for row in rows:
        tags = [str(tag).lower() for tag in (row.get('tags') or [])]
        text = (str(row.get('title', '')) + ' ' + str(row.get('ticker', ''))).lower()
        if ('tennis' in tags or 'tennis' in text) and not (
            'table tennis' in tags or any(word in text for word in ('table tennis', 'tabletennis', 'pickleball'))
        ) and row.get('ticker'):
            found.append(row)
    return sorted(found, key=lambda row: row.get('title', row['ticker']))


def tennis_quote(market, side, kind):
    try:
        value = market.get(f'{side}_{kind}_dollars')
        price = float(value) if value is not None else float(market[f'{side}_{kind}']) / 100
        return price if math.isfinite(price) and 0 < price < 1 else None
    except (KeyError, TypeError, ValueError):
        return None


def show_tennis():
    st.header('Kalshi tennis')
    st.caption('Open tennis contracts · public market data · paper research')
    st.info('YES and NO prices are market quotes, not model win probabilities. Open contracts may include matches already in progress. Check the rules and match status before paper testing.')
    try:
        if st.button('Refresh tennis catalog and prices', key='tennis_refresh'):
            tennis_catalog.clear()
            tt_open.clear()
        catalog = tennis_catalog()
        if not catalog:
            st.info('No tennis series are available in the current catalog. Try refreshing later.')
            return
        mapping = {row['ticker']: row for row in catalog}
        default = next((i for i, row in enumerate(catalog) if row['ticker'] == 'KXATPMATCH'), 0)
        series = st.selectbox('Tennis market series', list(mapping), index=default,
                              format_func=lambda ticker: mapping[ticker].get('title', ticker) + ' · ' + ticker,
                              key='tennis_series')
        poll = st.checkbox('Refresh tennis prices every minute while open', value=True, key='tennis_poll')
        @st.fragment(run_every=60 if poll else None)
        def board():
            try:
                markets, limited, fetched = tt_open(series)
                stale = pd.Timestamp.now(tz='UTC') - pd.Timestamp(fetched) > pd.Timedelta(seconds=90)
                st.caption('Snapshot: ' + pd.Timestamp(fetched).tz_convert('America/New_York').strftime('%b %d, %I:%M:%S %p %Z'))
                if stale:
                    st.warning('Snapshot is stale. Refresh prices before using these quotes.')
                if limited:
                    st.warning('Page limit reached; some contracts may be missing.')
                if not markets:
                    st.info('No open contracts for this series. Choose another tennis series.')
                    return
                search = st.text_input('Find player or match', key='tennis_search').strip().lower()
                selected = [m for m in markets if not search or search in ' '.join(str(m.get(k) or '') for k in ('title', 'subtitle', 'yes_sub_title', 'event_ticker', 'ticker')).lower()]
                if not selected:
                    st.info('No contracts match that search.')
                    return
                rows = []
                for m in selected:
                    row = {'Contract': m.get('title') or m.get('ticker'), 'YES selection': m.get('yes_sub_title', ''),
                           'Status': m.get('status', ''), 'Ticker': m.get('ticker', ''),
                           'Close time (not match start)': m.get('close_time', '')}
                    for side in ('yes', 'no'):
                        for kind in ('bid', 'ask'):
                            value = tennis_quote(m, side, kind)
                            row[f'{side.upper()} {kind}'] = f'{value * 100:.1f}¢' if value is not None else 'Unavailable'
                    rows.append(row)
                frame = pd.DataFrame(rows)
                st.dataframe(frame, hide_index=True, use_container_width=True)
                choices = {m['ticker']: m for m in selected if m.get('ticker')}
                if not choices:
                    return
                ticker = st.selectbox('Inspect tennis contract', list(choices),
                                      format_func=lambda t: (choices[t].get('title') or t) + ' · ' + t,
                                      key='tennis_inspect_' + series)
                market = choices[ticker]
                st.write('YES selection:', market.get('yes_sub_title') or 'See contract rules')
                st.code(ticker, language=None)
                with st.expander('Settlement and retirement rules', expanded=True):
                    st.write(market.get('rules_primary') or 'Rules unavailable; verify on Kalshi.')
                    st.write(market.get('rules_secondary') or '')
                export = frame.copy()
                export['Fetched at (UTC)'] = fetched
                st.download_button('Download tennis market snapshot', export.to_csv(index=False),
                                   'kalshi_tennis_markets.csv', 'text/csv', key='tennis_export_' + series)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                st.warning(f'Tennis market feed paused: {exc}')
        board()
    except (ValueError, KeyError, TypeError) as exc:
        st.warning(f'Tennis catalog unavailable: {exc}')


def show_table_tennis():
    import altair as alt
    st.header("Table tennis · match winners")
    mode = st.radio("Data mode", ["Automatic Kalshi scanner", "Manual DraftKings / Kalshi"], horizontal=True, key="tt_data_mode")
    if mode == "Automatic Kalshi scanner":
        show_table_tennis_auto()
        return
    st.caption("Pregame research for DraftKings and Kalshi · manual quotes · paper testing")
    st.info("Upload completed matches from one league. Elo estimates relative strength; its percentages are experimental and have not been calibrated. No live odds connection or automatic bets are enabled.")
    st.download_button("Download history CSV template", "completed_at,player_a,player_b,winner\n", "table_tennis_history_template.csv", "text/csv")
    upload = st.file_uploader("Completed match history CSV", type="csv", key="tt_history")
    source = st.text_input("History source / league", placeholder="Name the source and league you verified")
    a_col, b_col = st.columns(2)
    a = a_col.text_input("Player A — exact history name").strip()
    b = b_col.text_input("Player B — exact history name").strip()
    start = st.text_input("Upcoming match start (include time zone)", placeholder="2026-11-02T20:00:00-05:00")
    book = st.selectbox("Platform", ["DraftKings", "Kalshi"], key="tt_book")
    if book == "DraftKings":
        ca, cb = st.columns(2)
        qa = ca.number_input("Player A American odds", value=-110, step=5)
        qb = cb.number_input("Player B American odds", value=-110, step=5)
    else:
        st.caption("Use only a verified match-winner contract. YES must mean Player A wins and NO must mean Player B wins. Enter executable ask prices, not last-traded prices.")
        contract = st.text_input("Kalshi contract ticker / URL")
        verified = st.checkbox("I verified the players, start time, YES/NO mapping, and settlement rules")
        ca, cb = st.columns(2)
        qa = ca.number_input("YES ask — cents", 1, 99, 50)
        qb = cb.number_input("NO ask — cents", 1, 99, 50)
    fee = st.number_input("Estimated fee per $1 payout — cents", 0.0, 99.0, 0.0, step=0.1)
    st.caption("Enter fees from the platform's current rules. Zero means fees excluded; this is not a fee schedule.")
    quote_time = st.text_input("Odds captured at (include time zone)", placeholder="2026-11-02T19:58:00-05:00")
    verified_quote = st.checkbox("These are current pregame quotes for this exact match")
    if not st.button("Analyze match", key="tt_analyze"):
        return
    try:
        if upload is None or not source.strip() or not a or not b or a == b:
            raise ValueError("Upload history, identify its source, and enter two distinct player names.")
        cutoff, quoted = pd.Timestamp(start), pd.Timestamp(quote_time)
        if cutoff.tzinfo is None or quoted.tzinfo is None:
            raise ValueError("Start and quote timestamps must include a time zone.")
        cutoff, quoted = cutoff.tz_convert("UTC"), quoted.tz_convert("UTC")
        now = pd.Timestamp.now(tz="UTC")
        if cutoff <= now or quoted > now or now - quoted > pd.Timedelta(minutes=5) or not verified_quote:
            raise ValueError("Use an upcoming match and verified odds captured within the last five minutes.")
        if book == "Kalshi" and (not contract.strip() or not verified):
            raise ValueError("Provide and verify an actual Kalshi match-winner contract. Availability is not assumed.")
        ratings, counts, h = tt_ratings(pd.read_csv(upload), min(cutoff, now))
        if min(counts.get(a, 0), counts.get(b, 0)) < 20:
            raise ValueError("WAIT — each player needs at least 20 completed matches in the uploaded league history. This is a research filter, not proof of accuracy.")
        pa = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
        if book == "DraftKings":
            da, db = tt_decimal(qa), tt_decimal(qb)
            costs = [1 / da, 1 / db]
        else:
            costs = [qa / 100, qb / 100]
        probs = [pa, 1 - pa]
        net_costs = [x + fee / 100 for x in costs]
        edges = [p - c for p, c in zip(probs, net_costs)]
        best = int(np.argmax(edges))
        names = [a, b]
        st.subheader("Research outcome")
        st.metric("Higher Elo win estimate", names[int(pa < 0.5)], f"{max(probs):.1%}")
        st.write("Paper-test candidate: " + names[best] if edges[best] >= 0.05 else "WAIT — neither side clears the 5 percentage-point research margin.")
        st.caption("A positive modeled margin is not a demonstrated betting edge. Validate future picks before relying on it.")
        frame = pd.DataFrame({"Player": names, "Elo estimate": probs, "Break-even incl. entered fee": net_costs, "Model margin": edges, "History matches": [counts[a], counts[b]]})
        long = frame.melt(id_vars="Player", value_vars=["Elo estimate", "Break-even incl. entered fee"], var_name="Measure", value_name="Probability")
        chart = alt.Chart(long).mark_bar().encode(
            y=alt.Y("Player:N", title=None), yOffset="Measure:N",
            x=alt.X("Probability:Q", title="Probability / break-even", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, max(1, max(net_costs))])),
            color=alt.Color("Measure:N", scale=alt.Scale(range=["#2563eb", "#ea580c"]), legend=alt.Legend(orient="bottom")),
            tooltip=["Player:N", "Measure:N", alt.Tooltip("Probability:Q", format=".1%")]).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Blue: experimental Elo estimate. Orange: probability needed to break even at your entered price and fee.")
        display = frame.copy()
        for col in ["Elo estimate", "Break-even incl. entered fee", "Model margin"]:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        frame["platform"], frame["match_start"], frame["quoted_at"] = book, cutoff.isoformat(), quoted.isoformat()
        frame["history_source"], frame["result"] = source, "PENDING"
        st.download_button("Save this paper pick", frame.to_csv(index=False), "table_tennis_paper_pick.csv", "text/csv")
        st.caption("Save the CSV before leaving. This analysis uses your uploaded history and manually entered quotes; it does not poll either platform.")
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        st.warning(str(exc))


def main():
    st.set_page_config(page_title="Trout Sports Lab", page_icon="🏟️", layout="wide")
    st.markdown("""<style>
    .block-container {max-width:1100px;padding-top:1.5rem}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px}
    .stButton>button {min-height:48px;font-weight:600}
    [data-testid="stSidebar"] {border-right:1px solid #64748b33}
    h1 {letter-spacing:-1.4px} h2 {letter-spacing:-.5px}
    div[data-testid="stMetric"] {border:1px solid #3b4758;border-radius:12px;padding:16px}
    .stButton>button {border-radius:8px}
    </style>""", unsafe_allow_html=True)
    st.title("Trout Sports Lab")
    st.caption("Find a sport. Compare the odds. Understand your pick.")
    pages = ["All-sports leaderboard", "Sport sections", "Kalshi tennis", "Table tennis", "Paper journal", "BTC contract tracking", "Kalshi all crypto", "BTC trading"]
    labels = dict(zip(pages, ["🏠 Top picks", "🏟️ All sports", "🎾 Kalshi tennis", "🏓 Table tennis", "📒 Practice picks", "₿ BTC tracking", "🪙 Crypto markets", "📈 BTC research"]))
    section = st.sidebar.radio("Where would you like to go?", pages, format_func=lambda p: labels[p], key="research_area")
    st.sidebar.caption("Prices and market estimates, not guaranteed predictions. Export journals before closing: sessions can reset.")
    if section in ("All-sports leaderboard", "Sport sections"):
        sports_workspace(section == "All-sports leaderboard")
        return
    if section == "Paper journal":
        show_sports_journal(odds_key())
        return
    if section == "Kalshi tennis":
        show_tennis()
        return
    if section == "Table tennis":
        show_table_tennis()
        return
    if section == "BTC contract tracking":
        show_btc_tracking()
        return
    if section == "Kalshi all crypto":
        show_crypto_scanner()
        return
    _kc_render()
    with st.sidebar:
        interval = st.selectbox("Signal timeframe", ["15m", "1h"])
        max_daily = st.number_input("Max research signals/day", 1, 20, 4)
        refresh = st.button("Refresh BTC data")
        st.caption("The signal limit is counted by New York calendar day.")
    if refresh:
        fetch.clear()
    try:
        with st.spinner("Loading BTC candles..."):
            raw = fetch(interval)
        if raw is None or raw.empty:
            st.error("No data returned. Try Refresh BTC data.")
            return
        closed = closed_candles(raw, interval)
        data = indicators(closed)
        if len(data) < 70:
            st.error("Not enough completed candles to evaluate the strategy.")
            return
    except Exception as exc:
        st.error(f"BTC data unavailable: {type(exc).__name__}. Try refreshing later.")
        return
    duration = pd.Timedelta(minutes=15) if interval == "15m" else pd.Timedelta(hours=1)
    end = data.index[-1] + duration
    stale = pd.Timestamp.now(tz="UTC") - end > duration + pd.Timedelta(minutes=5)
    last = data.iloc[-1]
    current = "WAIT - STALE DATA" if stale else signal(last)
    cols = st.columns(4)
    cols[0].metric("Last completed BTC close", f"${last.Close:,.2f}")
    cols[1].metric("Indicator direction (before model gate)", current)
    cols[2].metric("RSI (14)", f"{last.RSI:.1f}")
    cols[3].metric("ATR (14)", f"${last.ATR:,.2f}")
    st.caption(f"Candle ended {end.tz_convert('America/New_York'):%Y-%m-%d %I:%M %p %Z}. "
               "Source: Yahoo Finance; refresh manually. Finalized data may still be revised.")
    if stale:
        st.warning("Data is stale. Current signal is disabled until fresh candles arrive.")
    live_panel = st.container()
    st.subheader("BTC price · New York time")
    visible = st.select_slider("Candles shown", options=[24, 48, 96, 240], value=48)
    averages = st.multiselect("Moving averages", ["EMA9", "EMA21", "EMA50"], default=["EMA21"])
    chart = data[["Close"] + averages].tail(visible).copy()
    chart.index = chart.index.tz_convert("America/New_York").tz_localize(None)
    import altair as alt
    plot = chart.rename_axis("New York time").reset_index().melt("New York time", var_name="Line", value_name="Price")
    st.altair_chart(alt.Chart(plot).mark_line(strokeWidth=2.5).encode(
        x=alt.X("New York time:T", title="New York time", axis=alt.Axis(format="%b %d %H:%M", labelAngle=-30, tickCount=5)),
        y=alt.Y("Price:Q", title="BTC price (USD)", axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False)),
        color=alt.Color("Line:N", scale=alt.Scale(domain=["Close", "EMA9", "EMA21", "EMA50"], range=["#2563eb", "#ea580c", "#9333ea", "#059669"]), legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("New York time:T", format="%b %d %H:%M"), "Line:N", alt.Tooltip("Price:Q", format="$,.2f")]
    ).properties(height=360), use_container_width=True)
    st.caption("Price uses a zoomed vertical scale. Close is the completed candle price; moving averages smooth the trend.")
    with st.expander("RSI · momentum guide"):
        rsi = data[["RSI"]].tail(visible).copy()
        rsi.index = rsi.index.tz_convert("America/New_York").tz_localize(None)
        rc = alt.Chart(rsi.rename_axis("New York time").reset_index()).mark_line(color="#2563eb").encode(
            x="New York time:T", y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100])), tooltip=["New York time:T", alt.Tooltip("RSI:Q", format=".1f")])
        rules = alt.Chart(pd.DataFrame({"Level": [30, 70]})).mark_rule(color="#ea580c", strokeDash=[5, 5]).encode(y="Level:Q")
        st.altair_chart((rc + rules).properties(height=180), use_container_width=True)
        st.caption("Dashed lines mark 30 and 70. RSI alone does not predict a reversal.")

    st.subheader("Historical direction test")
    st.info("Tests next candle close versus its open, with no invented contract prices or P/L. "
            "This is a BTC direction proxy; it does not reproduce Kalshi settlement rules.")
    trades = direction_test(data, interval, int(max_daily))
    if trades.empty:
        st.write("No qualifying signals.")
    else:
        sample = st.radio("Evaluation sample", ["Holdout", "Earlier", "All"], horizontal=True)
        selected = trades if sample == "All" else trades[trades['sample'] == sample]
        decisive = selected[selected.outcome != "TIE"]
        n = len(decisive)
        if n:
            wins = int(decisive.correct.sum())
            lo, hi = wilson(wins, n)
            cols = st.columns(3)
            cols[0].metric("Non-tie signals", n)
            cols[1].metric("Direction accuracy", f"{100 * wins / n:.1f}%")
            cols[2].metric("95% Wilson interval", f"{100 * lo:.1f} to {100 * hi:.1f}%")
            st.caption(f"Ties excluded: {len(selected) - n}. Always-UP baseline on these same "
                       f"candles: {100 * (decisive.outcome == 'UP').mean():.1f}%. "
                       "Intervals assume independent observations; clustered signals can make them too narrow.")
        else:
            st.write("No non-tie signals in this sample.")
        st.caption("Latest 30% is a chronological holdout, not a new untouched dataset if you already viewed "
                   "or tuned against it. Rules are fixed; no optimization or probability calibration is performed.")
        st.dataframe(selected.tail(100), use_container_width=True)
        st.download_button("Download direction research CSV", trades.to_csv(index=False),
                           f"btc_direction_{interval}.csv", "text/csv")

    report = show_model(data, interval, int(max_daily), stale)
    snapshot = live_snapshot(report, end, interval, int(max_daily),
                             st.session_state.setdefault("btc_forward_journal", []))
    with live_panel:
        st.subheader("Live BTC signal — manual simulation")
        cols = st.columns(3)
        cols[0].metric("Bot suggestion", snapshot["action"])
        p = snapshot["p_up"]
        cols[1].metric("Next-candle UP estimate", f"{100 * p:.1f}%" if p is not None else "Unavailable")
        cols[2].metric("Next-candle DOWN estimate", f"{100 * (1-p):.1f}%" if p is not None else "Unavailable")
        st.info(snapshot["reason"])
        finish = pd.Timestamp(snapshot["window_end"]).tz_convert("America/New_York")
        st.caption(f"Forecast candle ends {finish:%Y-%m-%d %I:%M %p %Z}. Estimates use completed Yahoo candles, "
                   "not a real-time tick feed. Refresh manually about 60–120 seconds after a candle closes. "
                   "SKIP means no qualifying bot entry. Model scores are experimental, not guaranteed win rates.")
    show_kalshi_quotes()
    show_forward_journal(snapshot, int(max_daily))
    st.subheader("Manual Kalshi contract paper calculator")
    st.caption("Enter the price of the exact side you would buy. Fees are TOTAL dollars for the order, "
               "copied from the platform. This does not fetch markets, submit orders, or determine settlement.")
    balance = st.number_input("Virtual contract budget ($)", min_value=0.0, value=100.0)
    ticker = st.text_input("Contract ticker / expiration (for your journal)")
    side = st.selectbox("Purchased side", ["UP / YES", "DOWN / NO"])
    price = st.number_input("Purchase price (cents per contract)", 1, 99, 50)
    qty = st.number_input("Number of contracts", 1, 10000, 1)
    entry_fee = st.number_input("Total entry fee ($)", min_value=0.0, value=0.0, step=0.01)
    outcome = st.selectbox("Hypothetical or observed result", ["WIN", "LOSS", "SELL"])
    exit_price, exit_fee = 0, 0.0
    if outcome == "SELL":
        exit_price = st.number_input("Sale price (cents per contract)", 0, 100, 50)
        exit_fee = st.number_input("Total sale fee ($)", min_value=0.0, value=0.0, step=0.01)
    cost, pnl = paper_pnl(price, qty, entry_fee, outcome, exit_price, exit_fee)
    cost_col, loss_col, pnl_col = st.columns(3)
    cost_col.metric("Entry cost with fee", f"${cost:.2f}")
    loss_col.metric("Maximum settlement loss", f"${cost:.2f}")
    pnl_col.metric(f"{outcome} paper result", f"${pnl:+.2f}")
    st.caption(f"Settlement break-even probability under these entered fees: {100 * cost / qty:.2f}%. "
               "Indicator accuracy above is not a calibrated probability for this contract.")
    if cost > balance:
        st.warning("Entry cost exceeds the virtual budget. Reduce quantity before logging.")
    st.caption("WIN/LOSS must follow the exact contract rules, reference price and expiration. "
               "Voids or refunds require a separate adjustment. This journal is manual and session-only; "
               "each row is an independent scenario, not a portfolio ledger.")
    if "journal" not in st.session_state:
        st.session_state.journal = []
    if st.button("Log paper scenario", disabled=cost > balance or not ticker.strip()):
        st.session_state.journal.append(dict(time=datetime.now(timezone.utc).isoformat(),
            contract=ticker, side=side, price_cents=price, quantity=qty,
            entry_fee=entry_fee, outcome=outcome, exit_cents=exit_price,
            exit_fee=exit_fee, entry_cost=cost, net_pnl=pnl))
    if st.session_state.journal:
        journal = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal, use_container_width=True)
        st.download_button("Save paper scenarios CSV", journal.to_csv(index=False),
                           "kalshi_paper_scenarios.csv", "text/csv")
    with st.expander("Indicators and testing limitations"):
        st.write("EMA 9/21/50 measure trend alignment; RSI filters momentum; volume must exceed its "
                 "20-candle average. ATR measures movement size, not direction. WAIT means rules do not align. "
                 "No historical contract quotes, bid/ask spreads, automatic paper execution, calibrated "
                 "contract probabilities or profitability claims are included. Forward testing on exact "
                 "contract quotes and settlements is still required.")



import re as _kc_re
from datetime import datetime as _kc_datetime, timezone as _kc_timezone
from urllib.parse import quote as _kc_quote
from zoneinfo import ZoneInfo as _kc_ZoneInfo
import requests as _kc_requests
import streamlit as _kc_st

_KC_BASE = "https://external-api.kalshi.com/trade-api/v2"

def _kc_get(path, params=None):
    url = _KC_BASE + path
    if params:
        url += "?" + urlencode(params)
    return get_json(url)[0]

@_kc_st.cache_data(ttl=600, show_spinner=False)
def _kc_series():
    series = _kc_get("/series").get("series", [])
    def is_crypto(item):
        metadata = str([item.get("category", ""), item.get("categories", []),
                        item.get("tags", [])]).lower()
        title = str(item.get("title", ""))
        return "crypto" in metadata or bool(_kc_re.search(
            r"\b(bitcoin|ethereum|solana|crypto|cryptocurrency|BTC|ETH|SOL|XRP|dogecoin)\b",
            title, _kc_re.I))
    return sorted([s for s in series if s.get("ticker") and is_crypto(s)],
                  key=lambda s: (str(s.get("title", "")), s["ticker"]))

@_kc_st.cache_data(ttl=20, show_spinner=False)
def _kc_markets(series_ticker):
    markets, cursor, seen = {}, None, set()
    for _ in range(20):
        params = {"series_ticker": series_ticker, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = _kc_get("/markets", params)
        for market in data.get("markets", []):
            if market.get("ticker"):
                markets[market["ticker"]] = market
        cursor = data.get("cursor")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Kalshi repeated a pagination cursor. Please refresh.")
        seen.add(cursor)
    return (sorted(markets.values(), key=lambda m: (m.get("close_time") or "", m["ticker"])),
            bool(cursor), _kc_datetime.now(_kc_timezone.utc).isoformat())

@_kc_st.cache_data(ttl=20, show_spinner=False)
def _kc_detail(ticker):
    market = _kc_get("/markets/" + _kc_quote(ticker, safe="")).get("market")
    if not isinstance(market, dict):
        raise ValueError("Kalshi returned no contract details.")
    return market, _kc_datetime.now(_kc_timezone.utc).isoformat()

def _kc_time(value):
    if not value:
        return "Not provided"
    try:
        return _kc_datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            _kc_ZoneInfo("America/New_York")).strftime("%b %d, %Y %I:%M:%S %p %Z")
    except (ValueError, TypeError):
        return str(value)

def _kc_price(value):
    try:
        price = float(value)
        return f"{price * 100:.2f}¢" if 0 < price < 1 else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"

def _kc_render():
    _kc_st.subheader("Kalshi crypto contracts")
    _kc_st.caption("Live market snapshots • Read only • No orders")
    if not _kc_st.checkbox("Open crypto contract selector", key="kc_enabled"):
        return
    if _kc_st.button("Refresh crypto data", key="kc_refresh"):
        _kc_series.clear()
        _kc_markets.clear()
        _kc_detail.clear()
    try:
        with _kc_st.spinner("Finding crypto series…"):
            series = _kc_series()
        if not series:
            _kc_st.warning("No crypto series were discovered. Refresh and check again.")
            return
        by_series = {s["ticker"]: s for s in series}
        ticker = _kc_st.selectbox("Choose a crypto series", list(by_series),
            format_func=lambda t: f"{by_series[t].get('title', t)} — {t}", key="kc_series_choice")
        _kc_st.caption("A series groups recurring contracts. Look for 15-minute or hourly series in this list.")
        with _kc_st.spinner("Loading open contracts…"):
            markets, partial, fetched = _kc_markets(ticker)
        if partial:
            _kc_st.warning("Showing the first 20 pages; this series has more contracts.")
        _kc_st.caption(f"List retrieved: {_kc_time(fetched)}")
        if not markets:
            _kc_st.info("This series has no open contracts right now. Choose another series.")
            return
        _kc_st.dataframe([{
            "Contract": m.get("title"), "YES outcome": m.get("yes_sub_title"),
            "Ticker": m["ticker"], "Closes (Eastern)": _kc_time(m.get("close_time")),
            "YES ask": _kc_price(m.get("yes_ask_dollars")),
            "NO ask": _kc_price(m.get("no_ask_dollars")),
        } for m in markets], hide_index=True)
        by_market = {m["ticker"]: m for m in markets}
        selected = _kc_st.selectbox("Choose a contract to inspect", list(by_market),
            format_func=lambda t: " | ".join(str(v) for v in (
                by_market[t].get("title"), by_market[t].get("yes_sub_title"),
                _kc_time(by_market[t].get("close_time")), t) if v),
            key="kc_contract_" + ticker)
        market, retrieved = _kc_detail(selected)
        _kc_st.session_state["kalshi_selected_contract"] = {
            "series": by_series[ticker], "market": market, "retrieved_at": retrieved}
        _kc_st.markdown("**Selected contract**")
        _kc_st.write(market.get("title", selected))
        _kc_st.code(selected, language=None)
        _kc_st.write("Status:", market.get("status", "Not provided"))
        _kc_st.write("YES outcome:", market.get("yes_sub_title") or "See settlement rules")
        _kc_st.write("NO outcome:", market.get("no_sub_title") or "See settlement rules")
        _kc_st.write("Trading closes:", _kc_time(market.get("close_time")))
        _kc_st.write("Expected expiration:", _kc_time(market.get("expected_expiration_time")))
        _kc_st.write("Expiration:", _kc_time(market.get("expiration_time")))
        _kc_st.write("Strike type:", market.get("strike_type") or "See settlement rules")
        _kc_st.write("Lower / target strike:", market.get("floor_strike") if market.get("floor_strike") is not None else "Not provided")
        _kc_st.write("Upper strike:", market.get("cap_strike") if market.get("cap_strike") is not None else "Not provided")
        if market.get("functional_strike") or market.get("custom_strike"):
            _kc_st.write("Additional target information:", market.get("functional_strike"), market.get("custom_strike"))
        _kc_st.write("YES bid / ask:", _kc_price(market.get("yes_bid_dollars")), "/", _kc_price(market.get("yes_ask_dollars")))
        _kc_st.write("NO bid / ask:", _kc_price(market.get("no_bid_dollars")), "/", _kc_price(market.get("no_ask_dollars")))
        _kc_st.caption(f"Details retrieved: {_kc_time(retrieved)}. Prices exclude fees and do not guarantee a fill. Use Refresh for a new snapshot.")
        with _kc_st.expander("Settlement rules and source", expanded=True):
            _kc_st.write(market.get("rules_primary") or "Primary rules not provided by API.")
            if market.get("rules_secondary"):
                _kc_st.write(market["rules_secondary"])
            for source in by_series[ticker].get("settlement_sources", []):
                _kc_st.write(source.get("name", "Settlement source"), source.get("url", ""))
            for field in ("contract_url", "contract_terms_url"):
                if by_series[ticker].get(field):
                    _kc_st.write(by_series[ticker][field])
        _kc_st.info("Contract selected for inspection. Your existing price-direction model is not yet connected to this contract; no contract probability is calculated here.")
    except (FeedError, _kc_requests.RequestException, ValueError, KeyError, TypeError) as error:
        _kc_st.session_state.pop("kalshi_selected_contract", None)
        _kc_st.error(f"Could not load Kalshi crypto data: {error}")




def odds_key():
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def active_sports(api_key):
    catalog, _, _ = get_json("https://api.the-odds-api.com/v4/sports?" + urlencode({"apiKey": api_key}))
    return sorted([r for r in catalog if r.get("active") and r.get("key")], key=lambda r: (r.get("group", "Other"), r.get("title", "")))


def eligible_quotes(rows):
    if rows.empty:
        return rows.copy()
    now = pd.Timestamp.now(tz="UTC")
    starts = pd.to_datetime(rows.start, utc=True, errors="coerce")
    times = pd.to_datetime(rows.quote_time, utc=True, errors="coerce")
    return rows.loc[(starts > now) & ((now-times).dt.total_seconds().between(0, 600))].copy()


def render_parlays(rows):
    st.subheader("Build a practice parlay")
    st.caption("Every pick must win. Adding picks usually lowers the chance of winning.")
    rows = eligible_quotes(rows)
    if not rows.empty:
        rows = rows.loc[rows.market != "outrights"]
    if rows.empty:
        st.info("No fresh pregame legs. Refresh odds to build a research ticket.")
        return
    # All legs must be offered at the same book; no synthetic cross-book ticket.
    book = st.selectbox("Ticket sportsbook", sorted(rows.book.unique()), key="parlay_book")
    pool = rows[rows.book == book].dropna(subset=["normalized_market_pct"]).sort_values("normalized_market_pct", ascending=False)
    pool = pool.drop_duplicates("event_id").reset_index(drop=True)
    st.caption("One market-favored candidate per game. Same-game combinations are excluded. Book acceptance must be checked separately.")
    if len(pool) < 2:
        st.info("At least two different games with complete market prices are needed.")
        return
    ids = st.multiselect("Choose 2–4 legs", list(pool.index), default=list(pool.index[:2]), max_selections=4,
        format_func=lambda i: f"{pool.iloc[i].game} | {pool.iloc[i].selection} {pool.iloc[i].line} | {pool.iloc[i].decimal_odds:.2f}")
    if len(ids) < 2:
        return
    legs = pool.loc[ids]
    st.dataframe(legs[["game", "market", "selection", "line", "decimal_odds", "normalized_market_pct"]], hide_index=True, use_container_width=True)
    price = float(legs.decimal_odds.prod())
    estimate = float((legs.normalized_market_pct / 100).prod())
    c1, c2 = st.columns(2)
    c1.metric("Indicative combined decimal odds", f"{price:.2f}")
    c2.metric("Estimated chance if legs are independent", f"{estimate:.1%}")
    st.caption("The probability multiplies margin-adjusted book estimates under an independence assumption. Cross-game correlation and model error can change the actual chance. Combined prices are illustrative and exclude promotions.")
    st.download_button("Export research ticket", legs.to_csv(index=False), "parlay_research.csv", "text/csv")



def simple_pick_cards(rows, sort_by_chance=True):
    st.subheader("Picks at a glance")
    st.caption("Best available prices. Market chances come from prices, not a proven prediction model.")
    if rows.empty:
        st.info("No fresh upcoming picks here yet.")
        return
    ordered = rows.sort_values("normalized_market_pct", ascending=False) if sort_by_chance else rows
    candidates = ordered.drop_duplicates("event_id").head(6)
    for _, row in candidates.iterrows():
        with st.container(border=True):
            st.caption(str(row.game))
            line = "" if pd.isna(row.line) else f" {row.line:g}"
            market = {"h2h": "Winner", "spreads": "Point spread", "totals": "Total score", "outrights": "Season winner"}.get(row.market, row.market)
            player = f"{row.player} " if row.player else ""
            st.subheader(f"{player}{row.selection}{line}")
            st.write(f"{market} · {row.book}")
            chance = row.normalized_market_pct
            a, b = st.columns(2)
            a.metric("Market chance", "Not available" if pd.isna(chance) else f"{chance:.0f}%")
            b.metric("Odds", f"{int(row.american_odds):+d}")
            st.caption(f"A $1 practice stake returns ${row.decimal_odds:.2f} total if it wins, including the $1 stake.")
            start = pd.to_datetime(row.start, utc=True, errors="coerce")
            if pd.notna(start):
                st.caption("Starts " + start.tz_convert("America/New_York").strftime("%b %d · %I:%M %p %Z"))
            with st.expander("Why is this shown?"):
                st.write("This is the first selection for this game under your current sorting. The displayed book has the best available payout for this exact selection in this scan.")
                st.caption("Chances may be conditional on no tie or push. Injuries and lineups are not checked. Prices can change.")


def sports_workspace(combined):
    key = odds_key()
    st.header("Your sports dashboard" if combined else "Find your sport")
    st.write("Choose a sport, compare picks, and see what the numbers mean.")
    with st.expander("New here? Start with these three words"):
        st.write("**Odds:** the payout offered. **Market chance:** an estimate from sportsbook prices. **Parlay:** several picks joined together; every leg must win, subject to the book’s rules.")
        st.caption("A bigger payout usually means a lower chance. No pick is a sure win.")
    if not key:
        st.info("Add ODDS_API_KEY to Streamlit Secrets to load the sports catalog and current sportsbook odds.")
        st.code('ODDS_API_KEY = "YOUR_KEY"', language="toml")
        st.write("The sidebar preserves your Kalshi and BTC research tools. No sportsbook data or picks are invented while the feed is disconnected.")
        return
    try:
        catalog = active_sports(key)
    except (FeedError, ValueError, TypeError):
        st.error("Could not load the sports catalog. Check your key and feed access.")
        return
    if not catalog:
        st.info("No active sports returned by the provider.")
        return
    by_key = {r['key']: r for r in catalog}
    st.caption(f"{len(catalog)} active competitions across {len({r.get('group', 'Other') for r in catalog})} sport categories • supplied by your feed")
    with st.expander("See all available sports and leagues"):
        st.dataframe(pd.DataFrame([{"Sport": r.get('group', 'Other'), "League / competition": r.get('title', r['key']), "Type": 'Season winner' if r.get('has_outrights') else 'Game odds'} for r in catalog]), hide_index=True, use_container_width=True)
    search = st.text_input("Find a sport or league", placeholder="Try soccer, basketball, tennis…").strip().lower()
    visible = [r for r in catalog if not search or search in (str(r.get('group', '')) + ' ' + str(r.get('title', '')) + ' ' + r['key']).lower()]
    if not visible:
        st.info("No match in the current feed. Try another name.")
        return
    if not combined:
        group = st.selectbox("Sport category", sorted({r.get('group', 'Other') for r in visible}))
        leagues = [r['key'] for r in visible if r.get('group', 'Other') == group]
        selected = st.selectbox("League / competition", leagues, format_func=lambda k: by_key[k]['title'])
        show_sports(by_key[selected]['title'], selected, 'outrights' if by_key[selected].get('has_outrights') else 'h2h,spreads,totals')
        return
    selected = st.multiselect("Leagues to scan", [r["key"] for r in visible], default=[k for k in ('americanfootball_nfl', 'basketball_nba', 'icehockey_nhl') if k in {r['key'] for r in visible}], format_func=lambda k: by_key[k]['title'])
    if st.checkbox("Include every league shown above", value=False):
        selected = [r["key"] for r in visible]
    st.caption("Scan only the leagues you want. Each league uses feed credits; snapshots are cached for five minutes.")
    if st.button("Scan selected leagues", type="primary", disabled=not selected):
        frames, errors = [], []
        for league in selected:
            try:
                events, quota, fetched = sports_feed(league, key, 'outrights' if by_key[league].get('has_outrights') else 'h2h,spreads,totals')
                frame = odds_rows(events, fetched)
                frames.append(frame)
            except (FeedError, ValueError, TypeError):
                errors.append(by_key[league]['title'])
        st.session_state['leaderboard'] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.session_state['scan_errors'] = errors
    for error in st.session_state.get('scan_errors', []):
        st.warning(f"{error}: feed unavailable for this scan.")
    raw = st.session_state.get('leaderboard', pd.DataFrame())
    if raw.empty:
        st.info("Choose leagues and scan to see available upcoming games.")
        return
    rows = eligible_quotes(raw)
    if selected and not rows.empty:
        rows = rows[rows.sport.isin(selected)]
    else:
        rows = rows.iloc[:0]
    if rows.empty:
        st.info("No fresh upcoming quotes remain. Scan again.")
        return
    books = st.multiselect("Sportsbooks", sorted(rows.book.unique()), default=sorted(rows.book.unique()))
    rows = rows[rows.book.isin(books)]
    rank = st.radio("Rank by", ["Market-estimated likelihood", "Best payout"], horizontal=True)
    best = best_quotes(rows).copy()
    best = best.sort_values('normalized_market_pct' if rank == 'Market-estimated likelihood' else 'decimal_odds', ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Upcoming games", rows.event_id.nunique())
    c2.metric("Sportsbooks", rows.book.nunique())
    c3.metric("Leagues", rows.sport.nunique())
    simple_pick_cards(best, sort_by_chance=False)
    with st.expander("Compare all prices and details"):
        st.dataframe(best[["sport", "game", "start", "market", "selection", "line", "book", "american_odds", "normalized_market_pct", "quote_time"]], hide_index=True, use_container_width=True)
    st.caption("Percentages are margin-adjusted sportsbook estimates where complete outcome prices exist. They are not validated forecasts. Injury reports and starting lineups are not verified. Best payout ranks price, not likelihood or expected profit.")
    st.download_button("Export leaderboard", best.to_csv(index=False), "sports_leaderboard.csv", "text/csv")
    render_parlays(rows)


if __name__ == "__main__":
    main()
