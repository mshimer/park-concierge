"""Fill the vector store from the National Park Service API, or from the committed snapshot.

    uv run park-ingest              # live if NPS_API_KEY is set, else offline
    uv run park-ingest --offline    # always the committed snapshot
    uv run park-ingest --rebuild    # drop and re-embed even if the collection looks current

The snapshot exists so the demo, the eval suite and CI are reproducible and runnable with no NPS
key and no network. Live mode overwrites it, so a refresh is one command.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chroma import open_collection, park_document, park_metadata
from .config import settings

log = logging.getLogger(__name__)


NPS_BASE = "https://developer.nps.gov/api/v1"
PAGE_SIZE = 50
TIMEOUT = httpx.Timeout(30.0)

# Elevation is not an NPS API field, but the demo needs it: a customer can carry a
# `max_elevation_ft` limit and the evals check the agent respects it. Curated once, merged into
# both the live and offline paths so the two behave identically.
ELEVATIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "park_elevations.json"

# NOTE : reminder to comprehend this Python Comprehension - it's faster than .map() and more readable than a for loop. 
#        This normalizes one park at a time into a flat dictionary
def _normalize(park: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """NPS field names -> the flat snake_case record the rest of the project speaks.
    
        NOTE: changes the dict keys to snake_case, and flattens the structure of the park record.
    """
    return {
        "park_code": park["parkCode"],
        "name": park["fullName"],
        "states": park["states"],
        "designation": park.get("designation") or "National Park System unit",
        "description": park.get("description", ""),
        "weather_info": park.get("weatherInfo", ""),
        "activities": [a["name"] for a in park.get("activities", [])],
        "topics": [t["name"] for t in park.get("topics", [])],
        "lat": float(park["latitude"]) if park.get("latitude") else None,
        "lon": float(park["longitude"]) if park.get("longitude") else None,
        "alerts": [
            {
                "title": a.get("title", ""),
                "category": a.get("category", ""),
                "description": a.get("description", ""),
            }
            for a in alerts
        ],
    }


def fetch_live(api_key: str, designation: str = "National Park") -> list[dict[str, Any]]:

    """Page through /parks, then pull /alerts for the ones we kept."""
    
    headers = {"X-Api-Key": api_key}
    parks: list[dict[str, Any]] = []
    with httpx.Client(base_url=NPS_BASE, headers=headers, timeout=TIMEOUT) as client:
        start = 0
        while True:
            r = client.get("/parks", params={"limit": PAGE_SIZE, "start": start})
            if r.status_code == 429:
                raise SystemExit("NPS rate limit (1000/hour) exhausted; try again later")
            r.raise_for_status()
            body = r.json()
            batch = body["data"]
            parks.extend(p for p in batch if p.get("designation") == designation)
            start += len(batch)
            log.info("fetched %s/%s parks", start, body["total"])
            if start >= int(body["total"]) or not batch:
                break

        # One alerts call for every park at once - the endpoint takes a comma-separated parkCode.
        codes = ",".join(p["parkCode"] for p in parks)
        r = client.get("/alerts", params={"parkCode": codes, "limit": 500})
        r.raise_for_status()
        by_code: dict[str, list[dict[str, Any]]] = {}
        for alert in r.json()["data"]:
            by_code.setdefault(alert["parkCode"], []).append(alert)

    return [_normalize(p, by_code.get(p["parkCode"], [])) for p in parks]


def _merge_elevations(parks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ELEVATIONS_PATH.exists():
        log.warning("no %s; elevation-constrained scenarios will not work", ELEVATIONS_PATH.name)
        return parks
    elevations = json.loads(ELEVATIONS_PATH.read_text())
    for park in parks:
        park.update(elevations.get(park["park_code"], {}))
    return parks


def load_parks(*, offline: bool) -> list[dict[str, Any]]:
    """Live when a NPS_API_KEY is available and offline was not forced; snapshot otherwise."""
    if not offline and settings.nps_api_key:
        parks = fetch_live(settings.nps_api_key)
        settings.snapshot_path.write_text(json.dumps(parks, indent=2, sort_keys=True) + "\n")
        log.info("wrote %s parks to %s", len(parks), settings.snapshot_path.name)
    else:
        if not offline:
            log.info("NPS_API_KEY not set; using the committed snapshot")
        parks = json.loads(settings.snapshot_path.read_text())
    return _merge_elevations(parks)


def ingest(
    *,
    offline: bool = False,
    rebuild: bool = False,
    chroma_dir: str | None = None,
    embeddings: Embeddings | None = None,
) -> int:
    """Embed every park into the Chroma collection. Returns the chunk count.

    `embeddings` is injectable so the fast test suite can build a collection with
    `HashEmbeddings` - no OpenAI key, no spend, deterministic ordering.
    """
    parks = load_parks(offline=offline)

    # RecursiveCharacterTextSplitter tries natural break points—such as paragraphs, 
    # then lines, then spaces—before splitting more abruptly
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for park in parks:
        meta = park_metadata(park)
        for i, chunk in enumerate(splitter.split_text(park_document(park))):
            texts.append(chunk)
            metadatas.append({**meta, "chunk": i})
            # The hash makes the id change when the text does, so edited chunks get re-embedded.
            digest = hashlib.sha256(chunk.encode()).hexdigest()[:12]
            ids.append(f"{park['park_code']}:{i}:{digest}")

    store = open_collection(chroma_dir=chroma_dir, embeddings=embeddings)
    if rebuild and store._collection.count():
        log.info("dropping %s existing chunks", store._collection.count())
        store.reset_collection()

    # Sync by id: drop chunks that no longer exist, embed only the ones that are new.
    # Embeddings cost money; re-running the demo should not re-buy the same vectors.
    existing_ids = set(store.get()["ids"])
    stale = existing_ids - set(ids)
    if stale:
        log.info("removing %s stale chunks", len(stale))
        store.delete(ids=sorted(stale))

    new = [n for n, id_ in enumerate(ids) if id_ not in existing_ids]
    if not new:
        log.info("collection already holds %s chunks; nothing to do (use --rebuild to force)", len(ids))
        return len(ids)

    store.add_texts(
        texts=[texts[n] for n in new],
        metadatas=[metadatas[n] for n in new],
        ids=[ids[n] for n in new],
    )
    log.info("embedded %s new chunks (%s total) from %s parks", len(new), len(ids), len(parks))
    return len(ids)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="ignore NPS_API_KEY, use the snapshot")
    ap.add_argument("--rebuild", action="store_true", help="re-embed even if the count matches")
    args = ap.parse_args()
    count = ingest(offline=args.offline, rebuild=args.rebuild)
    print(f"{count} chunks in collection {settings.collection!r} at {settings.chroma_dir}", file=sys.stderr)
