r"""Archetype discovery with an autoencoder (deliverable #1).

Idea
----
Each player-season becomes a "style profile": their per-game averages across the
raw stat categories (kicks, marks, tackles, hitouts, goals, ...). We train an
autoencoder to squeeze that ~14-number profile down through a 2D bottleneck and
reconstruct it. To reconstruct well from just 2 numbers, the network must learn
the few axes that actually separate playing styles. We then cluster that 2D
latent space to surface archetypes (ruck, inside mid, rebounding defender, ...).

This is a task linear models and trees can't do: it's about *representation*, not
prediction. (An autoencoder with no hidden layers / linear activations would
recover PCA; the nonlinearity lets it bend the space to fit the data better.)

Outputs:
  data/processed/player_archetypes.csv     (latent coords + cluster per profile)
  reports/figures/archetype_latent_space.png
  reports/figures/archetype_profiles.png

Run:
    .\.venv\Scripts\python.exe src\autoencoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
OUT = ROOT / "data" / "processed" / "player_archetypes.csv"
FIG_DIR = ROOT / "reports" / "figures"

# Playing-style features (per-game averages). We omit 'disposals' (= kicks +
# handballs, redundant) and the fantasy/supercoach scores (derived, not style).
STYLE_FEATURES = [
    "kicks", "handballs", "marks", "goals", "behinds", "tackles", "hitouts",
    "goal_assists", "inside50s", "clearances", "clangers", "rebound50s",
    "frees_for", "frees_against",
]
MIN_GAMES = 8       # a season profile needs enough games to be stable
N_CLUSTERS = 6      # roughly: ruck, inside mid, winger, key fwd, small, defender
LATENT_DIM = 2

# A few recognisable players to annotate on the map (latest season if present).
ANNOTATE = [
    "Max Gawn", "Marcus Bontempelli", "Nick Daicos", "Charlie Curnow",
    "Jeremy Cameron", "Rory Laird", "Sam Walsh", "Harry Sheezel",
]


def build_profiles() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    games = df.groupby(["player", "season"]).size().rename("games")
    profiles = df.groupby(["player", "season"])[STYLE_FEATURES].mean()
    profiles = profiles.join(games).reset_index()
    profiles = profiles[profiles["games"] >= MIN_GAMES].reset_index(drop=True)
    print(f"player-season profiles (>= {MIN_GAMES} games): {len(profiles):,}")
    return profiles


def build_autoencoder(n_features: int) -> tuple[keras.Model, keras.Model]:
    inp = keras.layers.Input((n_features,))
    x = keras.layers.Dense(8, activation="relu")(inp)
    latent = keras.layers.Dense(LATENT_DIM, name="latent")(x)  # the bottleneck
    x = keras.layers.Dense(8, activation="relu")(latent)
    out = keras.layers.Dense(n_features)(x)

    autoencoder = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, latent, name="encoder")  # shares the trained weights
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def name_archetypes(profiles: pd.DataFrame) -> dict[int, str]:
    """Label each cluster by its most distinctive per-game stats (vs league avg)."""
    league_mean = profiles[STYLE_FEATURES].mean()
    league_std = profiles[STYLE_FEATURES].std()
    labels = {}
    for c in sorted(profiles["cluster"].unique()):
        centroid = profiles.loc[profiles["cluster"] == c, STYLE_FEATURES].mean()
        z = ((centroid - league_mean) / league_std).sort_values(ascending=False)
        top = ", ".join(f"{stat}+" for stat in z.head(2).index)
        labels[c] = top
    return labels


def main() -> None:
    profiles = build_profiles()

    X_raw = profiles[STYLE_FEATURES].to_numpy()
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    autoencoder, encoder = build_autoencoder(X.shape[1])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    autoencoder.fit(X, X, validation_split=0.15, epochs=400, batch_size=32, verbose=0, callbacks=[stop])

    recon = autoencoder.predict(X, verbose=0)
    recon_mse = float(np.mean((X - recon) ** 2))
    print(f"reconstruction MSE (standardized): {recon_mse:.3f}  "
          f"(0 = perfect, 1.0 = no better than predicting the mean)")

    latent = encoder.predict(X, verbose=0)
    profiles["latent_1"] = latent[:, 0]
    profiles["latent_2"] = latent[:, 1]

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    profiles["cluster"] = km.fit_predict(latent)
    labels = name_archetypes(profiles)

    print("\nArchetypes (cluster: most distinctive per-game stats):")
    for c, lab in labels.items():
        n = int((profiles["cluster"] == c).sum())
        print(f"  cluster {c} (n={n:4d}): {lab}")

    profiles["archetype"] = profiles["cluster"].map(labels)
    profiles.to_csv(OUT, index=False)
    print(f"\nwrote -> {OUT.relative_to(ROOT)}")

    # ---- Figure 1: the latent archetype map ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(profiles["latent_1"], profiles["latent_2"],
                     c=profiles["cluster"], cmap="tab10", s=12, alpha=0.5)
    plt.colorbar(sc, label="cluster")

    latest = profiles.sort_values("season").drop_duplicates("player", keep="last")
    for name in ANNOTATE:
        row = latest[latest["player"] == name]
        if not row.empty:
            x, y = row["latent_1"].iloc[0], row["latent_2"].iloc[0]
            plt.scatter([x], [y], color="black", s=30, zorder=5)
            plt.annotate(name, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Player-season archetype map (autoencoder 2D latent space)")
    plt.xlabel("latent dimension 1")
    plt.ylabel("latent dimension 2")
    plt.tight_layout()
    f1 = FIG_DIR / "archetype_latent_space.png"
    plt.savefig(f1, dpi=110)
    plt.close()

    # ---- Figure 2: archetype fingerprints (z-scored centroids) ----
    league_mean = profiles[STYLE_FEATURES].mean()
    league_std = profiles[STYLE_FEATURES].std()
    centroids = (
        profiles.groupby("cluster")[STYLE_FEATURES].mean()
        .sub(league_mean).div(league_std)
    )
    plt.figure(figsize=(11, 5))
    im = plt.imshow(centroids.to_numpy(), cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    plt.colorbar(im, label="std devs vs league average")
    plt.yticks(range(N_CLUSTERS), [f"{c}: {labels[c]}" for c in range(N_CLUSTERS)])
    plt.xticks(range(len(STYLE_FEATURES)), STYLE_FEATURES, rotation=45, ha="right")
    plt.title("Archetype fingerprints (per-game stats relative to league average)")
    plt.tight_layout()
    f2 = FIG_DIR / "archetype_profiles.png"
    plt.savefig(f2, dpi=110)
    plt.close()

    print(f"saved figures:\n  {f1.relative_to(ROOT)}\n  {f2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
