from typing import TypedDict, List, Dict


class RAGState(TypedDict):

    question: str

    context: str

    sources: List[Dict]

    answer: str
