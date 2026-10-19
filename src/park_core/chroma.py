"""The one place that opens the `parks` collection.

Ingest writes it, the MCP parks toolset reads it. 
If either side picked its own collection name or embedding model, 
searches would silently return nothing - a Chroma collection embedded by one model
is meaningless to another. One factory, both callers.
"""

import hashlib
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import settings


def open_collection(
    chroma_dir: str | None = None,
    collection: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open (or create) the persistent parks collection."""
    return Chroma(
        collection_name=collection or settings.collection,
        persist_directory=chroma_dir or settings.chroma_dir,
        embedding_function=embeddings or default_embeddings(),  # injection point for tests or prod usage
                                                                # I like this pattern for injection - it's clean and concise
    )


def default_embeddings() -> Embeddings:
    """The real thing. Imported lazily so the fast test suite never needs an OpenAI key."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embed_model)


class HashEmbeddings(Embeddings):
    """Deterministic offline embeddings for the fast test suite.

    Inherits from Embeddings and overrides methods to provide deterministic offline embeddings 
    for testing purposes using a hash.

    Not semantic - it hashes word tokens into a fixed-width bag-of-words vector. Two texts sharing
    vocabulary land near each other, which is enough to assert that retrieval is wired up and that
    results come back in a stable order, without spending a cent or needing a key. Anything that
    actually depends on *meaning* belongs behind the `eval` marker with real embeddings.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(h[:4], "big") % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # overriding Embeddings method
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    # overriding Embeddings method
    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


def park_document(park: dict[str, Any]) -> str:
    """Flatten one park record into the text that gets embedded.

    Activities and topics are folded into the text on purpose: they are what make parks cluster by
    likeness ("desert", "canyoneering", "tidepooling"), which is what `parks_similar_to` shows off.

    NOTE : Makes a list then joins it into a string for one park
    """
    parts = [
        park["name"],
        f"{park['designation']} in {park['states']}",
        park["description"],
        "Activities: " + ", ".join(park.get("activities", [])),
        "Topics: " + ", ".join(park.get("topics", [])),
    ]
    if park.get("weather_info"):
        parts.append("Weather: " + park["weather_info"])
    # Elevation goes into the embedded text as well as the metadata. A visitor asking for
    # "somewhere low and warm" is expressing a soft preference, and the text is what serves that;
    # a hard ceiling is a filter instead (see parks_search).
    if park.get("elevation_ft_min") is not None:
        parts.append(
            f"Elevation ranges from {park['elevation_ft_min']} to {park['elevation_ft_max']} feet."
        )
    return "\n".join(p for p in parts if p)


def park_metadata(park: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata must be scalars, so list fields are joined for `$contains`-style filtering.
     
       NOTE : Makes a dictionary for one park 
    """
    return {
        "park_code": park["park_code"],
        "name": park["name"],
        "states": park["states"],
        "designation": park["designation"],
        "activities": ", ".join(park.get("activities", [])),
        "alert_count": len(park.get("alerts", [])),
        # Numeric, so parks_search can apply a real $lte filter rather than hoping the embedding
        # understood "under 5000 feet".
        "elevation_ft_min": park.get("elevation_ft_min", -1),
        "elevation_ft_max": park.get("elevation_ft_max", -1),
    }
