"""Plain data classes shared by the backend modules."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A piece of extracted text with where it came from (page or time range)."""
    text: str
    page: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Chunk:
    text: str
    page: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class ProcessedSource:
    """A source after text extraction and chunking, before indexing."""
    document_id: str
    filename: str
    source_type: str                 # pdf | document | youtube | audio
    chunks: List[Chunk]
    size_bytes: int = 0
    page_count: int = 0
    video_id: Optional[str] = None
    duration: float = 0.0


@dataclass
class SourceInfo:
    """What the UI needs to know about an indexed source."""
    document_id: str
    filename: str
    source_type: str
    chunk_count: int
    size_bytes: int = 0
    page_count: int = 0
    video_id: Optional[str] = None
    duration: float = 0.0


@dataclass
class Evidence:
    """One retrieved passage. 'label' (S1, S2, ...) is how the LLM refers to it."""
    label: str
    text: str
    distance: float
    document_id: str
    filename: str
    source_type: str
    chunk_id: str
    page: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    video_id: Optional[str] = None


@dataclass
class SourceFinding:
    label: str
    finding: str
    evidence: Optional[Evidence] = None


@dataclass
class ConflictClaim:
    """One statement quoted from the evidence. Verified: the quote really occurs in that passage."""
    label: str
    quote: str
    evidence: Optional[Evidence] = None


@dataclass
class ConflictReport:
    status: str = "none"             # none | contradiction | different_perspectives
    explanation: str = ""
    differences: List[str] = field(default_factory=list)
    conclusion: str = ""
    claims: List[ConflictClaim] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    sufficient: bool = True
    findings: List[SourceFinding] = field(default_factory=list)
    conflict: ConflictReport = field(default_factory=ConflictReport)
    evidence: List[Evidence] = field(default_factory=list)
    structured: bool = True
    mode: str = "document"           # document (from sources) | general (model knowledge) | not_found
