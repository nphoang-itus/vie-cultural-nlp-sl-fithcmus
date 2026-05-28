from __future__ import annotations

from src.rag.context_builder import build_rag_context
from src.rag.prompt_builder import build_text_qa_rag_prompt
from src.rag.rag_service import RagQAInput, RagService
from src.rag.retriever import CulturalKnowledgeRetriever
from src.rag.vector_store import VectorSearchResult


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                doc_id="good",
                text="Bánh chưng là món ăn truyền thống trong dịp Tết.",
                metadata={"keyword": "bánh chưng", "category": "am_thuc"},
                distance=0.2,
                score=0.8,
            ),
            VectorSearchResult(
                doc_id="bad",
                text="Nội dung không liên quan.",
                metadata={"keyword": "không liên quan", "category": "khac"},
                distance=0.9,
                score=0.1,
            ),
        ]


def test_retriever_filters_low_score_contexts() -> None:
    retriever = CulturalKnowledgeRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        default_top_k=5,
        score_threshold=0.35,
    )

    contexts = retriever.search("Bánh chưng có ý nghĩa gì?", top_k=5)

    assert [context.doc_id for context in contexts] == ["good"]


def test_rag_context_is_empty_when_no_result_passes_threshold() -> None:
    retriever = CulturalKnowledgeRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        default_top_k=5,
        score_threshold=0.95,
    )

    result = build_rag_context(
        retriever,
        "bạn có thể giúp được gì tôi?",
        top_k=5,
    )

    assert result.contexts == []
    assert result.context_block == ""


def test_prompt_without_context_uses_no_evidence_fallback() -> None:
    prompt = build_text_qa_rag_prompt(
        question="bạn có thể giúp được gì tôi?",
        rag_context="",
    )

    assert "Không có ngữ cảnh văn hóa đủ liên quan" in prompt
    assert "Không suy đoán hoặc bịa chi tiết văn hóa" in prompt
    assert "không có ảnh đầu vào" in prompt
    assert "Ngữ cảnh văn hóa:" not in prompt


def test_general_tet_query_prefers_general_festival_context() -> None:
    contexts = [
        VectorSearchResult(
            doc_id="mut",
            text="Mứt Tết là món ăn ngày Tết.",
            metadata={"keyword": "mứt Tết", "category": "am_thuc"},
            distance=0.547,
            score=0.453,
        ),
        VectorSearchResult(
            doc_id="tet",
            text="Tết Nguyên Đán là lễ hội truyền thống quan trọng ở Việt Nam.",
            metadata={"keyword": "Tết Nguyên Đán", "category": "le_hoi"},
            distance=0.559,
            score=0.441,
        ),
    ]

    class TetVectorStore:
        def query(
            self,
            query_embedding: list[float],
            *,
            top_k: int = 3,
            where: dict | None = None,
        ) -> list[VectorSearchResult]:
            return contexts

    retriever = CulturalKnowledgeRetriever(
        embedder=FakeEmbedder(),
        vector_store=TetVectorStore(),
        default_top_k=5,
        score_threshold=0.35,
    )

    result = build_rag_context(
        retriever,
        "tôi muốn tìm hiểu về tết ở Việt Nam",
        top_k=5,
    )

    assert [context.doc_id for context in result.contexts] == ["tet", "mut"]


def test_explicit_filters_are_applied_even_when_default_filter_flags_are_off() -> None:
    retriever = CulturalKnowledgeRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        default_top_k=5,
        score_threshold=0.35,
    )
    service = RagService(
        retriever=retriever,
        top_k=5,
        use_category_filter=False,
    )

    output = service.prepare_prompt(
        RagQAInput(
            question="bánh bao",
            category="am_thuc",
            keyword="bánh bao",
        )
    )

    assert output.filters == {
        "$and": [
            {"category": "am_thuc"},
            {"keyword": "bánh bao"},
        ]
    }
