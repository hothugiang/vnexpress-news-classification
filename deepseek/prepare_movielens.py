"""
prepare_movielens.py
====================
Prepare MovieLens data for UH-CSG integration.

This script:
  1. Loads item_ids.json + entity2id.json from a CRS dataset (ReDial / INSPIRED)
  2. Parses DBpedia URLs to extract clean movie titles + release years
  3. Matches those against MovieLens movies.csv (title + year exact match,
     with title-only fallback)
  4. Filters MovieLens ratings to users who have ≥ min_items matched items
  5. Saves edges (user_local_idx → entity_id) as a .pt file

Usage:
  python prepare_movielens.py \
      --movielens_dir /path/to/ml-25m \
      --dataset_dir   /path/to/data \
      --dataset       redial \
      --output        movielens_edges_redial.pt

Design note (UH-CSG):
  Edge type E_UI connects MovieLens user nodes (V_U) to item nodes (V_I).
  V_U = ML users with ≥ 1 item that maps to a DBpedia entity in item_ids.json.
  The edges are stored as a [2, num_edges] int64 tensor:
    row 0 → user local index  (0 … K-1)
    row 1 → DBpedia entity ID (integer from entity2id.json)
"""

import argparse
import json
import os
import re
import csv
import torch
from collections import defaultdict
from loguru import logger


# ---------------------------------------------------------------------------
# 1.  DBpedia URL → (base_title, year)
# ---------------------------------------------------------------------------

# Suffixes we strip from the resource name before treating it as a title.
# Order matters: more specific patterns first.
_FILM_SUFFIX_PATTERNS = [
    # _(2019_film), _(2019_American_film), _(2019_British_animated_film), …
    r"_\(\d{4}[^)]*film\)",
    # _(film), _(animated_film), _(short_film), …
    r"_\([^)]*film\)",
    # _(2019), _(2019_TV_series), … — only strip when it's year-like
    r"_\(\d{4}[^)]*\)",
]
_FILM_SUFFIX_RE = re.compile("|".join(_FILM_SUFFIX_PATTERNS), re.IGNORECASE)

# Capture the 4-digit year inside any parenthesised suffix.
_YEAR_RE = re.compile(r"\((\d{4})[^)]*\)")


def _parse_dbpedia_url(url: str):
    """
    Parse a DBpedia entity URL into (base_title_lower, year_or_None).

    Examples
    --------
    "<http://dbpedia.org/resource/Beautiful_Creatures_(2013_film)>"
        → ("beautiful creatures", "2013")

    "<http://dbpedia.org/resource/Elizabeth_(film)>"
        → ("elizabeth", None)

    "<http://dbpedia.org/resource/Toy_Story>"
        → ("toy story", None)
    """
    # Strip angle brackets and get the resource slug after /resource/
    url = url.strip("<>")
    slug = url.split("/resource/")[-1]  # e.g. Beautiful_Creatures_(2013_film)

    # Extract year from slug before stripping suffixes
    year_match = _YEAR_RE.search(slug)
    year = year_match.group(1) if year_match else None

    # Strip film-related suffixes
    clean = _FILM_SUFFIX_RE.sub("", slug)

    # Replace underscores with spaces and normalise whitespace
    clean = clean.replace("_", " ").strip()

    return clean.lower(), year


# ---------------------------------------------------------------------------
# 2.  MovieLens title → (base_title, year)
# ---------------------------------------------------------------------------

_ML_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_ml_title(title: str):
    """
    Parse a MovieLens movie title into (base_title_lower, year_or_None).

    MovieLens titles look like:
        "Toy Story (1995)"
        "Jumanji (1995)"
        "Beautiful Creatures (2013)"
        "Doctor Zhivago (Doktor Zhivago) (1965)"   ← two parenthesised groups
    """
    title = title.strip()
    year_match = _ML_YEAR_RE.search(title)
    year = year_match.group(1) if year_match else None

    # Remove the trailing (YYYY) to get the base title
    if year_match:
        base = title[: year_match.start()].strip()
    else:
        base = title

    return base.lower(), year


# ---------------------------------------------------------------------------
# 3.  Build DBpedia entity → title lookup (restricted to item_ids)
# ---------------------------------------------------------------------------


def build_entity_title_lookup(dataset_dir: str, dataset: str):
    """
    Return two lookup structures built from entity2id.json + item_ids.json:

    title_year_to_eid : dict[(base_title_lower, year_str)] → entity_id
    title_only_to_eid : dict[base_title_lower] → entity_id
        (used as fallback when the DBpedia URL has no year)

    Only entities whose integer ID is in item_ids.json are included,
    because E_UI edges must link ML users to CRS *item* nodes only.
    """
    entity2id_path = os.path.join(dataset_dir, dataset, "entity2id.json")
    item_ids_path = os.path.join(dataset_dir, dataset, "item_ids.json")

    with open(entity2id_path, "r", encoding="utf-8") as f:
        entity2id: dict = json.load(f)  # {url_str: int_id}

    with open(item_ids_path, "r", encoding="utf-8") as f:
        item_ids_set: set = set(json.load(f))  # {int_id, …}

    logger.info(
        f"Loaded entity2id ({len(entity2id)} entries), "
        f"item_ids ({len(item_ids_set)} items)"
    )

    title_year_to_eid: dict = {}  # (title, year) → eid  — primary key
    title_only_to_eid: dict = {}  # title          → eid  — fallback

    for url, eid in entity2id.items():
        if eid not in item_ids_set:
            continue  # skip non-item entities (directors, genres, …)

        base_title, year = _parse_dbpedia_url(url)

        if not base_title:  # degenerate slug
            continue

        if year:
            key = (base_title, year)
            if key not in title_year_to_eid:
                title_year_to_eid[key] = eid
        # Always populate title-only lookup (year-less DBpedia entries
        # and as fallback for year-mismatch cases)
        if base_title not in title_only_to_eid:
            title_only_to_eid[base_title] = eid

    logger.info(
        f"Built title lookups: "
        f"{len(title_year_to_eid)} (title+year), "
        f"{len(title_only_to_eid)} (title only)"
    )
    return title_year_to_eid, title_only_to_eid


# ---------------------------------------------------------------------------
# 4.  Map MovieLens movieId → DBpedia entity_id
# ---------------------------------------------------------------------------


def build_ml_to_entity(
    movielens_dir: str,
    title_year_to_eid: dict,
    title_only_to_eid: dict,
):
    """
    Read MovieLens movies.csv and return:

    ml_to_entity : dict[ml_movieId_int] → entity_id_int

    Matching priority (for each ML movie):
      1. Exact (base_title, year)  — most reliable
      2. (base_title, year±1)      — handles off-by-one in encoding year
      3. base_title only           — last resort (higher false-positive risk)
    """
    movies_path = os.path.join(movielens_dir, "movies.csv")

    ml_to_entity: dict = {}
    stats = {"exact": 0, "year_off1": 0, "title_only": 0, "no_match": 0}

    with open(movies_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ml_id = int(row["movieId"])
            ml_title = row["title"]

            base, year = _parse_ml_title(ml_title)

            # --- Strategy 1: exact (title + year) ---
            if year and (base, year) in title_year_to_eid:
                ml_to_entity[ml_id] = title_year_to_eid[(base, year)]
                stats["exact"] += 1
                continue

            # --- Strategy 2: year ± 1 tolerance ---
            if year:
                for delta in (-1, 1):
                    adj_year = str(int(year) + delta)
                    if (base, adj_year) in title_year_to_eid:
                        ml_to_entity[ml_id] = title_year_to_eid[(base, adj_year)]
                        stats["year_off1"] += 1
                        break
                if ml_id in ml_to_entity:
                    continue

            # --- Strategy 3: title only (no year) ---
            if base in title_only_to_eid:
                ml_to_entity[ml_id] = title_only_to_eid[base]
                stats["title_only"] += 1
                continue

            stats["no_match"] += 1

    total = sum(stats.values())
    matched = total - stats["no_match"]
    logger.info(
        f"MovieLens title matching: {matched}/{total} matched "
        f"(exact={stats['exact']}, year±1={stats['year_off1']}, "
        f"title_only={stats['title_only']}, unmatched={stats['no_match']})"
    )
    return ml_to_entity


# ---------------------------------------------------------------------------
# 5.  Load ratings and build E_UI edge tensor
# ---------------------------------------------------------------------------


def load_ratings(movielens_dir: str, min_rating: float = 3.5):
    """
    Return dict[userId_int] → list[movieId_int]
    Only ratings ≥ min_rating are kept (positive interactions).
    """
    ratings_path = os.path.join(movielens_dir, "ratings.csv")
    user_movies: dict = defaultdict(list)
    total = kept = 0

    with open(ratings_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if float(row["rating"]) >= min_rating:
                user_movies[int(row["userId"])].append(int(row["movieId"]))
                kept += 1

    logger.info(
        f"Ratings: kept {kept}/{total} (≥{min_rating}) from {len(user_movies)} users"
    )
    return user_movies


def build_edges(
    user_movies: dict,
    ml_to_entity: dict,
    min_items_per_user: int = 3,
):
    """
    Build E_UI edges for UH-CSG.

    Only ML users who have ≥ min_items_per_user movies that map to a
    DBpedia entity are included (= V_U in the UH-CSG design).

    Returns
    -------
    edges : LongTensor [2, num_edges]
        row 0 — user local index   (contiguous 0 … K-1)
        row 1 — DBpedia entity_id  (integer from entity2id.json)
    num_users : int   (K)
    user_id_to_local : dict[ml_userId] → local_idx  (for debugging)
    """
    src, dst = [], []
    user_id_to_local: dict = {}
    local_idx = 0
    skipped = 0

    for user_id, ml_movies in user_movies.items():
        # Map ML movie IDs → entity IDs (keep only those with a mapping)
        entity_ids = [ml_to_entity[m] for m in ml_movies if m in ml_to_entity]
        if len(entity_ids) < min_items_per_user:
            skipped += 1
            continue

        user_id_to_local[user_id] = local_idx
        for eid in entity_ids:
            src.append(local_idx)
            dst.append(eid)
        local_idx += 1

    edges = torch.tensor([src, dst], dtype=torch.long)
    logger.info(
        f"E_UI edges: {local_idx} users (K), "
        f"{edges.shape[1]} edges, "
        f"{skipped} users skipped (< {min_items_per_user} mapped items)"
    )
    return edges, local_idx, user_id_to_local


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build MovieLens → DBpedia edges for UH-CSG E_UI"
    )
    parser.add_argument(
        "--movielens_dir",
        type=str,
        required=True,
        help="Path to MovieLens dataset dir (e.g. ml-25m/)",
    )
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument(
        "--dataset", type=str, default="redial", choices=["redial", "inspired"]
    )
    parser.add_argument("--output", type=str, default="movielens_edges.pt")
    parser.add_argument("--min_rating", type=float, default=3.5)
    parser.add_argument("--min_items_per_user", type=int, default=3)
    args = parser.parse_args()

    # Step 1 — Build entity title lookup (item entities only)
    title_year_to_eid, title_only_to_eid = build_entity_title_lookup(
        args.dataset_dir, args.dataset
    )

    # Step 2 — Map MovieLens movie IDs to DBpedia entity IDs
    ml_to_entity = build_ml_to_entity(
        args.movielens_dir, title_year_to_eid, title_only_to_eid
    )

    if not ml_to_entity:
        logger.error(
            "No matches found. Check that the dataset_dir/dataset path is correct "
            "and that entity2id.json contains DBpedia movie URLs."
        )
        return

    # Step 3 — Load ratings
    user_movies = load_ratings(args.movielens_dir, args.min_rating)

    # Step 4 — Build edge tensor
    edges, num_users, user_id_to_local = build_edges(
        user_movies, ml_to_entity, args.min_items_per_user
    )

    # Step 5 — Save
    payload = {
        "edges": edges,  # [2, E] int64
        "num_users": num_users,  # K (number of V_U nodes)
        "ml_to_entity": ml_to_entity,  # {ml_movieId: entity_id}
        "user_id_to_local": user_id_to_local,  # {ml_userId: local_idx}
    }
    torch.save(payload, args.output)
    logger.info(f"Saved to {args.output}  (K={num_users}, edges={edges.shape[1]})")


if __name__ == "__main__":
    main()
