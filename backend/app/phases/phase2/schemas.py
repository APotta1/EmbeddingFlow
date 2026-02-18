from pydantic import BaseModel, Field
from typing import Optional

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: Optional[str] = None
    source_api: str
    position: int
    domain: Optional[str] = None
    published_date: Optional[str] = None

class Phase2Output(BaseModel):
    original_query: str
    urls: list[SearchResult]
    total_searched: int
    queries_used: list[str]
