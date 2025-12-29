#!/usr/bin/env python3
"""
Verification script for Hybrid Search & Context Injection.

Tests:
1. Context injection is present in retrieved chunks
2. BM25 keyword matching works
3. Hybrid search fusion improves results
4. Specific queries that benefit from BM25 (team numbers, acronyms)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_processor import get_query_processor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_context_injection(chunks):
    """Verify that chunks contain context injection prefix."""
    issues = []
    for i, chunk in enumerate(chunks[:5]):  # Check top 5
        text = chunk.text
        if not text.startswith("[") or "\n" not in text:
            issues.append(
                f"Chunk {i+1} (ID: {chunk.chunk_id}) missing context prefix"
            )
        else:
            # Check if context contains expected fields
            context_line = text.split("\n")[0]
            if "Source:" not in context_line and "Section:" not in context_line:
                issues.append(
                    f"Chunk {i+1} context format unexpected: {context_line[:50]}"
                )
    
    return issues


def test_keyword_queries(processor):
    """Test queries that should benefit from BM25 keyword matching."""
    test_cases = [
        {
            "query": "254",
            "description": "Team number (exact match)",
            "expected_in_results": ["254"],
        },
        {
            "query": "NEO motor",
            "description": "Specific component name",
            "expected_in_results": ["NEO", "motor"],
        },
        {
            "query": "PDP",
            "description": "Acronym (Power Distribution Panel)",
            "expected_in_results": ["PDP", "Power Distribution"],
        },
        {
            "query": "2025 game pieces",
            "description": "Year + keyword combination",
            "expected_in_results": ["2025"],
        },
    ]
    
    results = []
    for test in test_cases:
        logger.info(f"\nTesting: {test['description']}")
        logger.info(f"Query: '{test['query']}'")
        
        response = processor.search(query=test["query"], limit=10)
        
        # Check context injection
        context_issues = check_context_injection(response.chunks)
        if context_issues:
            logger.warning(f"Context injection issues: {context_issues}")
        
        # Check if expected terms appear
        found_terms = []
        for chunk in response.chunks[:5]:
            text_lower = chunk.text.lower()
            for term in test["expected_in_results"]:
                if term.lower() in text_lower:
                    found_terms.append(term)
        
        # Log results
        logger.info(f"Retrieved {len(response.chunks)} chunks")
        logger.info(f"Found terms: {set(found_terms)}")
        logger.info(f"Expected terms: {set(test['expected_in_results'])}")
        
        if response.chunks:
            logger.info(f"Top result score: {response.chunks[0].score:.4f}")
            logger.info(f"Top result preview: {response.chunks[0].text[:150]}...")
        
        results.append({
            "test": test["description"],
            "query": test["query"],
            "chunks_retrieved": len(response.chunks),
            "found_terms": list(set(found_terms)),
            "expected_terms": test["expected_in_results"],
            "context_issues": context_issues,
            "top_score": response.chunks[0].score if response.chunks else 0.0,
        })
    
    return results


def test_hybrid_search_fusion(processor):
    """Test that hybrid search (Vector + BM25) is working."""
    logger.info("\n" + "="*60)
    logger.info("Testing Hybrid Search Fusion")
    logger.info("="*60)
    
    query = "drivetrain gear ratio"
    logger.info(f"Query: '{query}'")
    
    response = processor.search(query=query, limit=10)
    
    logger.info(f"\nRetrieved {len(response.chunks)} chunks")
    logger.info(f"Latency: {response.latency_ms:.2f}ms")
    
    # Check that we have results from both sources
    # (We can't directly check this, but we can verify scores are reasonable)
    if response.chunks:
        scores = [c.score for c in response.chunks]
        logger.info(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        logger.info(f"Score distribution: {len([s for s in scores if s > 0.01])} chunks with score > 0.01")
    
    # Check context injection
    context_issues = check_context_injection(response.chunks)
    if context_issues:
        logger.warning(f"Context injection issues: {context_issues}")
    else:
        logger.info("✓ Context injection verified in top results")
    
    # Show top 3 results
    logger.info("\nTop 3 Results:")
    for i, chunk in enumerate(response.chunks[:3], 1):
        logger.info(f"\n{i}. Score: {chunk.score:.4f}")
        logger.info(f"   Chunk ID: {chunk.chunk_id}")
        logger.info(f"   Year: {chunk.year}, Team: {chunk.team}, Binder: {chunk.binder}")
        logger.info(f"   Preview: {chunk.text[:200]}...")
    
    return {
        "chunks_retrieved": len(response.chunks),
        "latency_ms": response.latency_ms,
        "context_issues": context_issues,
        "top_scores": [c.score for c in response.chunks[:5]],
    }


def test_bm25_initialization(processor):
    """Verify BM25 index is initialized."""
    logger.info("\n" + "="*60)
    logger.info("Testing BM25 Initialization")
    logger.info("="*60)
    
    if processor.bm25 is None:
        logger.error("✗ BM25 index is not initialized!")
        return False
    
    if not processor.bm25_documents:
        logger.error("✗ BM25 documents list is empty!")
        return False
    
    logger.info(f"✓ BM25 index initialized with {len(processor.bm25_documents)} documents")
    
    # Test a simple BM25 search
    test_query = "test"
    bm25_results = processor._search_bm25(query=test_query, limit=5)
    logger.info(f"✓ BM25 search returned {len(bm25_results)} results for query '{test_query}'")
    
    return True


def main():
    """Run all verification tests."""
    logger.info("="*60)
    logger.info("Retrieval Quality Verification")
    logger.info("="*60)
    
    try:
        processor = get_query_processor()
        
        # Test BM25 initialization
        bm25_ok = test_bm25_initialization(processor)
        if not bm25_ok:
            logger.error("\n✗ BM25 initialization failed. Cannot proceed with tests.")
            return 1
        
        # Test hybrid search fusion
        fusion_results = test_hybrid_search_fusion(processor)
        
        # Test keyword queries
        keyword_results = test_keyword_queries(processor)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Summary")
        logger.info("="*60)
        
        logger.info(f"\nBM25 Initialization: {'✓ PASS' if bm25_ok else '✗ FAIL'}")
        logger.info(f"Hybrid Search: {fusion_results['chunks_retrieved']} chunks retrieved")
        logger.info(f"Context Injection Issues: {len(fusion_results['context_issues'])}")
        
        logger.info("\nKeyword Query Results:")
        for result in keyword_results:
            status = "✓" if result["chunks_retrieved"] > 0 else "✗"
            logger.info(
                f"  {status} {result['test']}: "
                f"{result['chunks_retrieved']} chunks, "
                f"found {len(result['found_terms'])}/{len(result['expected_terms'])} terms"
            )
        
        # Overall status
        all_passed = (
            bm25_ok and
            fusion_results["chunks_retrieved"] > 0 and
            len(fusion_results["context_issues"]) == 0
        )
        
        if all_passed:
            logger.info("\n✓ All critical tests passed!")
            return 0
        else:
            logger.warning("\n⚠ Some tests had issues. Review logs above.")
            return 1
            
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

