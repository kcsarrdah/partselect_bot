"""
RAG Service
Orchestrates the full RAG workflow: Retrieve → Augment → Generate
Combines vector search, prompt building, and LLM generation.
"""

import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from services.ingestion_pipeline import IngestionPipeline
from services.llm_service import LLMService
from prompts import build_rag_prompt, PARTSELECT_SYSTEM_PROMPT
from utils.logger import setup_logger, log_pipeline_step, log_success, log_error, log_warning, log_metric
from services.query_cache import QueryCache

# Setup logger
logger = setup_logger(__name__)


class RAGService:
    """Service for handling RAG queries end-to-end."""

    # Part number pattern (PS followed by digits)
    PART_NUMBER_PATTERN = r'\bPS\d+\b'
    
    # Repair keywords
    REPAIR_KEYWORDS = [
        'repair', 'fix', 'broken', 'not working', 'malfunction',
        'troubleshoot', 'issue', 'problem', 'error', 'faulty', 'damaged',
        'replace', 'installation', 'install', 'how to', 'guide'
    ]
    
    # Stock/availability keywords
    STOCK_KEYWORDS = [
        'in stock', 'available', 'availability', 'price', 'cost', 'buy',
        'purchase', 'order', 'stock', 'inventory', 'how much', 'pricing'
    ]
    
    # Brand keywords (common appliance brands)
    BRAND_KEYWORDS = [
        'whirlpool', 'ge', 'general electric', 'frigidaire', 'kenmore',
        'lg', 'samsung', 'maytag', 'kitchenaid', 'bosch', 'amana',
        'electrolux', 'hotpoint', 'magic chef', 'haier', 'admir'
    ]
    
    # Installation keywords
    INSTALLATION_KEYWORDS = [
        'install', 'installation', 'how to install', 'how can i install',
        'how do i install', 'installing', 'setup', 'set up'
    ]

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        llm_service: LLMService,
        system_prompt: str = PARTSELECT_SYSTEM_PROMPT,
        default_k: int = 5,
        enable_cache: bool = True,
        enable_retrieval_logging: bool = True
    ):
        """
        Initialize RAG service with dependencies.
        
        Args:
            ingestion_pipeline: For vector search
            llm_service: For LLM generation
            system_prompt: Default system prompt
            default_k: Default number of docs to retrieve
            enable_cache: Enable query caching
            enable_retrieval_logging: Enable retrieval quality logging
        """
        self.pipeline = ingestion_pipeline
        self.llm = llm_service
        self.system_prompt = system_prompt
        self.default_k = default_k

        # Initialize query cache
        self.cache_enabled = enable_cache
        if self.cache_enabled:
            self.cache = QueryCache()
            log_success(logger, "Query cache enabled")
        else:
            self.cache = None
            log_warning(logger, "Query cache disabled")
        
        # Retrieval quality logging
        self.retrieval_logging_enabled = enable_retrieval_logging
        self.retrieval_logs = []  # Store logs in memory (move to DB in production)
        
        # Stats tracking
        self.queries_processed = 0
        self.total_response_time = 0.0
        
        log_success(logger, "RAG Service initialized")
    
    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        filter_type: Optional[str] = None,
        include_examples: bool = True,  # Changed default to True
        use_self_consistency: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query end-to-end with RAG.
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve (default: 5)
            filter_type: Filter by doc type ("part", "blog", "repair")
            include_examples: Include few-shot examples in prompt
            use_self_consistency: Enable self-consistency for critical queries
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        if self.cache_enabled:
            cached = self.cache.get(user_query)
            if cached:
                self.queries_processed += 1
                response_time = time.time() - start_time
                tokens_saved = cached['metadata'].get('tokens_used', 0)
                log_success(logger, f"Cache hit! (saved {tokens_saved} tokens)")
                cached['metadata']['cache_response_time_seconds'] = round(response_time, 3)
                
                try:
                    empty_context = []
                    cached['answer'] = self._embed_source_links(cached['answer'], empty_context)
                except Exception as e:
                    logger.error(f"Error post-processing cached answer: {e}")
                
                log_success(logger, f"Query Complete from Cache ({response_time:.3f}s)")
                return cached
        analysis = self._analyze_query(user_query)
        
        if not self._should_retrieve(user_query, analysis):
            result = self._answer_without_retrieval(user_query, analysis, start_time)
            if self.cache_enabled:
                self.cache.set(user_query, result)
            return result
        
        # Determine if we should use self-consistency
        is_critical_query = (
            len(analysis.get('part_numbers', [])) > 0 or
            analysis.get('intent') == 'installation'
        )
        
        if use_self_consistency and is_critical_query:
            return self._query_with_self_consistency(user_query, k, filter_type, include_examples, analysis)
        
        # Smart k selection
        if k is None:
            if analysis['part_numbers']:
                k = 3
            elif analysis['intent'] == 'stock':
                k = 3
            else:
                k = self.default_k
        
        try:
            log_pipeline_step(logger, 1, "Retrieving context")
            context_docs = self._retrieve_context(user_query, k, filter_type)
            
            if not context_docs:
                return self._handle_no_context(user_query)
            
            log_success(logger, f"Retrieved {len(context_docs)} documents")
            
            # CRITICAL FIX: If appliance not detected from query, extract it from retrieved part metadata
            if not analysis.get('appliance') and analysis.get('part_numbers'):
                for doc in context_docs:
                    doc_metadata = doc.get('metadata', {})
                    if doc_metadata.get('type') == 'part' and doc_metadata.get('appliance'):
                        analysis['appliance'] = doc_metadata.get('appliance')
                        logger.info(f"Detected appliance '{analysis['appliance']}' from part metadata")
                        break
            
            if analysis.get('part_numbers') and not analysis.get('is_repair_query') and not analysis.get('is_installation_query'):
                requested_parts = set(analysis['part_numbers'])
                filtered_docs = []
                for doc in context_docs:
                    doc_metadata = doc.get('metadata', {})
                    if doc_metadata.get('type') != 'part':
                        filtered_docs.append(doc)
                    elif doc_metadata.get('part_id') in requested_parts:
                        filtered_docs.append(doc)
                
                if filtered_docs:
                    context_docs = filtered_docs
            
            log_pipeline_step(logger, 2, "Building prompt")
            prompt = self._build_prompt(user_query, context_docs, include_examples, analysis)
            
            log_pipeline_step(logger, 3, "Generating response")
            llm_result = self.llm.generate(prompt)
            
            if llm_result['status_code'] != 200:
                return self._handle_llm_error(llm_result)
            
            log_success(logger, "Response generated")
            
            raw_answer = llm_result['response']
            sanitized_answer = self._sanitize_response(raw_answer)
            processed_answer = self._embed_source_links(sanitized_answer, context_docs)
            if not processed_answer or processed_answer.strip() == "":
                logger.error("Processed answer is empty after all processing!")
                processed_answer = raw_answer if raw_answer else "I couldn't generate a response. Please try again."
            
            log_pipeline_step(logger, 4, "Extracting sources")
            sources = self._extract_sources(context_docs)
            log_success(logger, f"Extracted {len(sources)} sources")
            
            response_time = time.time() - start_time
            self.queries_processed += 1
            self.total_response_time += response_time
            
            result = {
                "status_code": 200,
                "status": "success",
                "query": user_query,
                "answer": processed_answer,
                "sources": sources,
                "metadata": {
                    "retrieved_docs": len(context_docs),
                    "tokens_used": llm_result['usage']['total_tokens'],
                    "response_time_seconds": round(response_time, 2),
                    "model": llm_result['model'],
                    "filter_type": filter_type,
                    "method": "standard"
                }
            }
            
            if self.retrieval_logging_enabled:
                self._log_retrieval_quality(
                    query=user_query,
                    retrieved_docs=context_docs,
                    response=llm_result['response'],
                    analysis=analysis
                )
            
            if self.cache_enabled:
                self.cache.set(user_query, result)
            
            log_success(logger, f"Query Complete ({response_time:.2f}s, {llm_result['usage']['total_tokens']} tokens)")
            return result
        
        except Exception as e:
            log_error(logger, f"RAG Query failed: {e}")
            return {
                "status_code": 500,
                "status": "error",
                "query": user_query,
                "message": str(e),
                "error_type": type(e).__name__
            }
    
    
    def _normalize_appliance_name(self, query: str) -> str:
        """
        Normalize common misspellings of appliance names.
        
        Args:
            query: User query string
        
        Returns:
            Normalized query string
        """
        query_lower = query.lower()
        
        # Common misspellings mapping
        misspellings = {
            'refridgerator': 'refrigerator',
            'refridgerator': 'refrigerator',  # Common typo
            'refrigirator': 'refrigerator',
            'refrigrator': 'refrigerator',
            'refridgerator': 'refrigerator',
            'refrigerator': 'refrigerator',  # Correct spelling
            'fridge': 'refrigerator',  # Alias
            'freezer': 'freezer',
            'dishwasher': 'dishwasher',
            'dish washer': 'dishwasher',
            'dish-washer': 'dishwasher',
        }
        
        # Replace misspellings with correct spelling
        normalized = query_lower
        for misspelling, correct in misspellings.items():
            normalized = normalized.replace(misspelling, correct)
        
        return normalized
    
    
    def _should_retrieve(self, query: str, query_analysis: Dict) -> bool:
        """
        Determine if retrieval is necessary.
        
        Args:
            query: User query string
            query_analysis: Analysis results
        
        Returns:
            bool: True if retrieval needed, False otherwise
        """
        # Always retrieve for:
        if query_analysis.get('part_numbers'):
            return True
        if query_analysis.get('is_stock_query'):
            return True
        if query_analysis.get('is_repair_query'):
            return True
        if query_analysis.get('is_installation_query'):
            return True
        
        # Don't retrieve for general questions LM can answer
        general_keywords = ['how often', 'should i', 'is it normal', 'why', 'what is', 'tell me about']
        query_lower = query.lower()
        
        
        normalized_query = self._normalize_appliance_name(query)
        
        if any(kw in query_lower for kw in general_keywords):
            # Check if it's appliance-specific (then retrieve)
            appliance_keywords = ['refrigerator', 'dishwasher', 'fridge', 'freezer']
            
            if not any(app in normalized_query for app in appliance_keywords):
                return False
        
        return True  # Default: retrieve
    
    
    def _answer_without_retrieval(self, query: str, analysis: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """
        Answer general questions without retrieval using LLM's parametric knowledge.
        
        Args:
            query: User query
            analysis: Query analysis
            start_time: Start time for response time calculation
        
        Returns:
            Response dictionary
        """
        simple_prompt = f"""{self.system_prompt.strip()}

USER QUESTION: {query}

Please answer this general question to the best of your ability. If you need specific part information, let the user know to check PartSelect.com or ask for their model number."""
        
        llm_result = self.llm.generate(simple_prompt)
        
        if llm_result['status_code'] != 200:
            return self._handle_llm_error(llm_result)
        
        response_time = time.time() - start_time
        
        return {
            "status_code": 200,
            "status": "success",
            "query": query,
            "answer": llm_result['response'],
            "sources": [],
            "metadata": {
                "retrieved_docs": 0,
                "tokens_used": llm_result['usage']['total_tokens'],
                "model": llm_result['model'],
                "method": "parametric_only",
                "response_time_seconds": round(response_time, 2)
            }
        }
    
    
    def _log_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[Dict],
        response: str,
        analysis: Dict[str, Any],
        user_satisfaction: Optional[float] = None
    ) -> None:
        """
        Log retrieval performance for analysis.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            response: Generated response
            analysis: Query analysis
            user_satisfaction: Optional user satisfaction score
        """
        if not retrieved_docs:
            return
        
        # Calculate average relevance score
        scores = [d.get('score', 0.0) for d in retrieved_docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Get document types
        doc_types = [d.get('metadata', {}).get('type', 'unknown') for d in retrieved_docs]
        
        # Check context utilization (simple heuristic)
        context_utilized = self._check_context_usage(response, retrieved_docs)
        
        quality_metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': analysis.get('intent'),
            'num_docs_retrieved': len(retrieved_docs),
            'avg_relevance_score': round(avg_score, 3),
            'doc_types': doc_types,
            'response_length': len(response),
            'user_satisfaction': user_satisfaction,
            'context_utilization': context_utilized
        }
        
        # Store in memory (in production, send to analytics DB)
        self.retrieval_logs.append(quality_metrics)
        
        # Log to file/console for debugging
    
    def _check_context_usage(self, response: str, retrieved_docs: List[Dict]) -> str:
        """
        Check if retrieved context was actually used in the response.
        
        Args:
            response: Generated response
            retrieved_docs: Retrieved documents
        
        Returns:
            str: 'high', 'medium', 'low', or 'none'
        """
        # Count citations
        citations = re.findall(r'\[Source \d+\]', response)
        if len(citations) >= 2:
            return 'high'
        elif len(citations) == 1:
            return 'medium'
        
        # Check if part numbers from context appear in response
        context_part_numbers = set()
        for doc in retrieved_docs:
            part_id = doc.get('metadata', {}).get('part_id')
            if part_id:
                context_part_numbers.add(part_id)
        
        response_part_numbers = set(re.findall(self.PART_NUMBER_PATTERN, response, re.IGNORECASE))
        
        if context_part_numbers and response_part_numbers.intersection(context_part_numbers):
            return 'medium' if not citations else 'high'
        
        return 'low'
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract intent and key information.
        
        Args:
            query: User's query string
        
        Returns:
            Dict with query analysis results
        """
        query_lower = query.lower()
        
        
        normalized_query = self._normalize_appliance_name(query)
        
        # Extract part numbers
        part_numbers = re.findall(self.PART_NUMBER_PATTERN, query, re.IGNORECASE)
        
        # Extract brand names (use original query for brand detection)
        detected_brands = []
        for brand in self.BRAND_KEYWORDS:
            if brand in query_lower:
                detected_brands.append(brand.title())
        
        # Check for repair intent (use normalized query)
        is_repair_query = any(keyword in normalized_query for keyword in self.REPAIR_KEYWORDS)
        
        # Check for stock intent (use normalized query)
        is_stock_query = any(keyword in normalized_query for keyword in self.STOCK_KEYWORDS)
        
        # Check for installation intent (use normalized query)
        is_installation_query = any(keyword in normalized_query for keyword in self.INSTALLATION_KEYWORDS)
        
        # Determine primary intent
        if part_numbers and is_installation_query:
            intent = 'installation'
        elif part_numbers:
            intent = 'part_lookup'
        elif is_installation_query:
            intent = 'installation'
        elif is_repair_query:
            intent = 'repair'
        elif is_stock_query:
            intent = 'stock'
        else:
            intent = 'general'
        
        return {
            'part_numbers': part_numbers,
            'brands': detected_brands,
            'is_repair_query': is_repair_query,
            'is_stock_query': is_stock_query,
            'is_installation_query': is_installation_query,
            'intent': intent,
            'query': query  # Include original query for reference
        }

    
    def _retrieve_context(
        self,
        query: str,
        k: int,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with smart prioritization and conditional retrieval.
        """
        try:
            analysis = self._analyze_query(query)
            reformulated_query = self._reformulate_query(query, analysis)
            
            # CONDITIONAL RETRIEVAL: Choose strategy based on query complexity
            query_complexity = self._assess_query_complexity(query, analysis)
            
            if query_complexity == 'simple':
                return self._simple_retrieval(reformulated_query, k, filter_type, analysis)
            
            elif query_complexity == 'moderate':
                return self._moderate_retrieval(reformulated_query, k, filter_type, analysis)
            
            else:  # complex
                return self._complex_retrieval(reformulated_query, k, filter_type, analysis)
        
        except Exception as e:
            log_error(logger, f"Retrieval error: {e}")
            return []
    
    def _assess_query_complexity(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        Assess query complexity to determine retrieval strategy.
        
        Returns:
            'simple', 'moderate', or 'complex'
        """
        # Simple: Single part number lookup
        if analysis.get('part_numbers') and len(analysis.get('part_numbers', [])) == 1:
            if not analysis.get('is_repair_query') and not analysis.get('is_installation_query'):
                return 'simple'
        
        # Complex: Multiple issues, vague queries, or multi-step problems
        query_lower = query.lower()
        complexity_indicators = [
            'and', 'also', 'both', 'multiple', 'several',
            'not working', 'broken', 'leaking', 'not cooling'
        ]
        
        if any(indicator in query_lower for indicator in complexity_indicators):
            return 'complex'
        
        # Moderate: Everything else
        return 'moderate'

    def _simple_retrieval(
        self,
        query: str,
        k: int,
        filter_type: Optional[str],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simple retrieval: single pass, focused search."""
        results = []
        
        # For simple part number lookups
        if analysis.get('part_numbers'):
            part_id = analysis['part_numbers'][0]
            # Use part number directly, not reformulated query
            part_result = self.pipeline.search(
                query=part_id,
                k=5,
                filter_metadata={"type": "part"}
            )
            
            if part_result['status_code'] == 200:
                exact_matches = part_result.get('results', [])
                # Filter to exact part_id match
                exact_matches = [
                    doc for doc in exact_matches 
                    if doc.get('metadata', {}).get('part_id') == part_id
                ]
                
                if exact_matches:
                    return exact_matches[:k]
        
        # Fallback: general search with reformulated query
        general_result = self.pipeline.search(query, k=k)
        if general_result['status_code'] == 200:
            return general_result.get('results', [])[:k]
        
        return []

    def _moderate_retrieval(
        self,
        query: str,
        k: int,
        filter_type: Optional[str],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Moderate retrieval: multi-step but not iterative."""
        results = []
        
        # Priority 1: If part number detected, search for exact match first
        if analysis.get('part_numbers'):
            part_id = analysis['part_numbers'][0]
            part_result = self.pipeline.search(
                query=part_id,
                k=5,
                filter_metadata={"type": "part"}
            )
            
            if part_result['status_code'] == 200:
                exact_matches = part_result.get('results', [])
                exact_matches = [
                    doc for doc in exact_matches 
                    if doc.get('metadata', {}).get('part_id') == part_id
                ]
                
                if exact_matches:
                    results.extend(exact_matches[:1])
                    
                    # For installation queries, also get repair guides
                    if analysis.get('intent') == 'installation':
                        repair_result = self.pipeline.search_by_type(
                            query=f"install {part_id}",
                            doc_type='repair',
                            k=2
                        )
                        if repair_result['status_code'] == 200:
                            repair_results = repair_result.get('results', [])
                            results.extend(repair_results[:2])
                    
                    if len(results) >= k:
                        return results[:k]
        
        # Priority 2: Intent-based search
        if not filter_type:
            if analysis.get('intent') == 'repair':
                # Retrieve repair guides
                repair_result = self.pipeline.search_by_type(query, doc_type='repair', k=k//3 + 1)
                if repair_result['status_code'] == 200:
                    repair_results = repair_result.get('results', [])
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in repair_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
                
                
                blog_result = self.pipeline.search_by_type(query, doc_type='blog', k=k//3 + 1)
                if blog_result['status_code'] == 200:
                    blog_results = blog_result.get('results', [])
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in blog_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
                # Retrieve parts
                parts_query = query
                if analysis.get('brands'):
                    parts_query = f"{query} {analysis['brands'][0]}"
                
                parts_result = self.pipeline.search_by_type(parts_query, doc_type='part', k=k//3 + 1)
                if parts_result['status_code'] == 200:
                    parts_results = parts_result.get('results', [])
                    if analysis.get('brands'):
                        brand = analysis['brands'][0].lower()
                        parts_results = [p for p in parts_results if brand in p.get('metadata', {}).get('brand', '').lower()]
                    
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in parts_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
            
            elif analysis.get('intent') == 'stock':
                parts_result = self.pipeline.search_by_type(query, doc_type='part', k=k)
                if parts_result['status_code'] == 200:
                    parts_results = parts_result.get('results', [])
                    if analysis.get('brands'):
                        brand = analysis['brands'][0].lower()
                        parts_results = [p for p in parts_results if brand in p.get('metadata', {}).get('brand', '').lower()]
                    
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in parts_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
            
            elif analysis.get('intent') == 'installation':
                repair_result = self.pipeline.search_by_type(query, doc_type='repair', k=k//2 + 1)
                if repair_result['status_code'] == 200:
                    repair_results = repair_result.get('results', [])
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in repair_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
                
                parts_result = self.pipeline.search_by_type(query, doc_type='part', k=k//2 + 1)
                if parts_result['status_code'] == 200:
                    parts_results = parts_result.get('results', [])
                    # Prioritize parts with installation videos
                    parts_with_videos = [p for p in parts_results if p.get('metadata', {}).get('install_video_url')]
                    parts_without_videos = [p for p in parts_results if not p.get('metadata', {}).get('install_video_url')]
                    parts_results = parts_with_videos + parts_without_videos
                    
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in parts_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
        
        # Priority 3: General search
        if len(results) < k or filter_type:
            if filter_type:
                general_result = self.pipeline.search_by_type(query, doc_type=filter_type, k=k)
            else:
                general_result = self.pipeline.search(query, k=k)
            
            if general_result['status_code'] == 200:
                general_results = general_result.get('results', [])
                existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                for doc in general_results:
                    if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                        results.append(doc)
        
        # Re-rank: Put exact part matches first, then by score
        def sort_key(doc):
            metadata = doc.get('metadata', {})
            score = doc.get('score', 0.0)
            
            if analysis.get('part_numbers'):
                part_id = analysis['part_numbers'][0]
                if metadata.get('part_id') == part_id:
                    return (-1, -score)
            
            if analysis.get('brands'):
                brand = analysis['brands'][0].lower()
                if brand in metadata.get('brand', '').lower():
                    return (0, -score)
            
            return (1, -score)
        
        results.sort(key=sort_key)
        
        final_results = results[:k]
        return final_results

    def _complex_retrieval(
        self,
        query: str,
        k: int,
        filter_type: Optional[str],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Complex retrieval: iterative retrieval with reasoning feedback.
        Inspired by IRCoT (Trivedi et al., 2022).
        """
        results = []
        
        
        if analysis.get('intent') == 'repair' or analysis.get('is_repair_query'):
            blog_result = self.pipeline.search_by_type(query, doc_type='blog', k=k//3 + 1)
            if blog_result['status_code'] == 200:
                blog_results = blog_result.get('results', [])
                results.extend(blog_results)
        # Step 1: Initial broad retrieval
        initial_results = self.pipeline.search(query, k=k*2)
        if initial_results['status_code'] == 200:
            existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
            for doc in initial_results.get('results', []):
                if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                    results.append(doc)
        
        # Step 2: Analyze initial results to identify key concepts
        if results:
            key_concepts = self._extract_key_concepts(results[:3])
            
            # Step 3: Refined retrieval based on key concepts
            refined_query = f"{query} {key_concepts}"
            refined_results = self.pipeline.search(refined_query, k=k)
            
            if refined_results['status_code'] == 200:
                existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                for doc in refined_results.get('results', []):
                    if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                        results.append(doc)
        
        return results[:k]

    def _extract_key_concepts(self, docs: List[Dict[str, Any]]) -> str:
        """Extract key concepts from documents for query refinement."""
        concepts = []
        for doc in docs:
            metadata = doc.get('metadata', {})
            if metadata.get('part_name'):
                concepts.append(metadata['part_name'])
            if metadata.get('symptom'):
                concepts.append(metadata['symptom'])
        return ' '.join(concepts[:3])
    
    def _reformulate_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        Reformulate query with explicit intent to improve retrieval.
        """
        intent = analysis.get('intent', 'general')
        part_numbers = analysis.get('part_numbers', [])
        brands = analysis.get('brands', [])
        
        if intent == 'part_lookup' and part_numbers:
            part_nums = ', '.join(part_numbers)
            brand_text = f" for {brands[0]}" if brands else ""
            return f"Find part information for part number {part_nums}{brand_text}, including price, availability, installation instructions, and product details"
        
        if intent == 'installation':
            if part_numbers:
                part_nums = ', '.join(part_numbers)
                return f"Find installation instructions, installation videos, and installation steps for part {part_nums}"
            else:
                return f"Find installation guides, installation videos, and step-by-step installation instructions for: {query}"
        
        if intent == 'repair':
            brand_text = f" for {brands[0]}" if brands else ""
            return f"Find repair guides, troubleshooting steps, and replacement parts needed to fix: {query}{brand_text}"
        
        if intent == 'stock':
            brand_text = f" from {brands[0]}" if brands else ""
            return f"Find parts availability, stock status, and pricing information{brand_text} for: {query}"
        
        return f"Find relevant information about refrigerator and dishwasher parts, repair guides, or installation instructions for: {query}"
    
    def _build_prompt(
        self,
        user_query: str,
        context_docs: List[Dict[str, Any]],
        include_examples: bool = True,
        analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build complete RAG prompt with query-aware instructions.
        
        NOTE: This now uses the updated build_rag_prompt from prompts module
        which includes conflict detection, relevance ordering, and source citations.
        """
        if analysis is None:
            analysis = self._analyze_query(user_query)
        
        
        conversation_history = self.cache.get_conversation_history(limit=20) if self.cache_enabled else []
        
        
        return build_rag_prompt(
            user_query=user_query,
            context_docs=context_docs,
            system_prompt=self.system_prompt,
            include_examples=include_examples,
            query_analysis=analysis,
            conversation_history=conversation_history
        )
    
    def _extract_sources(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract clean source citations from context docs."""
        sources = []
        
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('type', 'unknown')
            
            source = {
                "type": doc_type,
                "score": doc.get('score', 0.0)
            }
            
            if doc_type == 'part':
                source.update({
                    "part_id": metadata.get('part_id'),
                    "part_name": metadata.get('part_name'),
                    "brand": metadata.get('brand'),
                    "price": metadata.get('price'),
                    "product_url": metadata.get('product_url'),
                    "install_video_url": metadata.get('install_video_url')
                })
            elif doc_type == 'repair':
                source.update({
                    "symptom": metadata.get('symptom'),
                    "appliance": metadata.get('appliance'),
                    "difficulty": metadata.get('difficulty'),
                    "video_url": metadata.get('video_url'),
                    "detail_url": metadata.get('detail_url')
                })
            elif doc_type == 'blog':
                source.update({
                    "title": metadata.get('title'),
                    "url": metadata.get('url')
                })
            
            sources.append(source)
        
        return sources
    
    def _handle_no_context(self, user_query: str) -> Dict[str, Any]:
        """Handle case where no relevant context is found."""
        log_warning(logger, "No relevant context found")
        
        return {
            "status_code": 404,
            "status": "no_context",
            "query": user_query,
            "answer": "I couldn't find relevant information in our database to answer your question. Please try rephrasing or contact PartSelect support for assistance.",
            "sources": [],
            "metadata": {
                "retrieved_docs": 0,
                "reason": "No relevant documents found"
            }
        }
    
    def _handle_llm_error(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM generation errors."""
        log_error(logger, f"LLM error: {llm_result.get('message')}")
        
        return {
            "status_code": llm_result['status_code'],
            "status": "llm_error",
            "message": llm_result.get('message', 'LLM generation failed'),
            "error_type": llm_result.get('error_type')
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        avg_response_time = (
            self.total_response_time / self.queries_processed
            if self.queries_processed > 0
            else 0.0
        )
     
        vector_stats = self.pipeline.get_status()
        
        stats = {
            "queries_processed": self.queries_processed,
            "average_response_time": round(avg_response_time, 2),
            "vector_store_docs": vector_stats.get('total_documents', 0),
            "llm_model": self.llm.model,
            "default_k": self.default_k
        }
        
        if self.cache_enabled and self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        
        if self.retrieval_logging_enabled and self.retrieval_logs:
            avg_relevance = sum(log['avg_relevance_score'] for log in self.retrieval_logs) / len(self.retrieval_logs)
            
            stats["retrieval_stats"] = {
                "total_logged": len(self.retrieval_logs),
                "avg_relevance_score": round(avg_relevance, 3),
                "context_utilization_breakdown": self._get_utilization_breakdown()
            }
        
        return stats
    
    def _get_utilization_breakdown(self) -> Dict[str, int]:
        """Get breakdown of context utilization levels."""
        breakdown = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        for log in self.retrieval_logs:
            util = log.get('context_utilization', 'none')
            breakdown[util] = breakdown.get(util, 0) + 1
        return breakdown
    
    def health_check(self) -> Dict[str, Any]:
        """Check if RAG system is healthy."""
        try:
            vector_status = self.pipeline.health_check()
            if vector_status['status_code'] != 200:
                return {
                    "status_code": 503,
                    "status": "unhealthy",
                    "reason": "Vector store unavailable"
                }
            
            llm_info = self.llm.get_model_info()
            if not llm_info.get('api_key_set'):
                return {
                    "status_code": 503,
                    "status": "unhealthy",
                    "reason": "LLM API key not set"
                }
            
            return {
                "status_code": 200,
                "status": "healthy",
                "vector_docs": vector_status.get('total_documents', 0),
                "llm_model": llm_info.get('model')
            }
        
        except Exception as e:
            return {
                "status_code": 500,
                "status": "error",
                "message": str(e)
            }

    def _query_with_self_consistency(
        self,
        user_query: str,
        k: Optional[int] = None,
        filter_type: Optional[str] = None,
        include_examples: bool = True,
        analysis: Optional[Dict[str, Any]] = None,
        num_samples: int = 3
    ) -> Dict[str, Any]:
        """Generate multiple responses and select the most consistent one."""
        start_time = time.time()
        
        if analysis is None:
            analysis = self._analyze_query(user_query)
        
        if k is None:
            if analysis['part_numbers']:
                k = 3
            elif analysis['intent'] == 'stock':
                k = 3
            else:
                k = self.default_k
        
        log_pipeline_step(logger, 1, "Retrieving context (shared across samples)")
        context_docs = self._retrieve_context(user_query, k, filter_type)
        
        if not context_docs:
            return self._handle_no_context(user_query)
        
        log_success(logger, f"Retrieved {len(context_docs)} documents")
        
        log_pipeline_step(logger, 2, "Building prompt")
        prompt = self._build_prompt(user_query, context_docs, include_examples, analysis)
        log_metric(logger, "Prompt size", f"{len(prompt)} chars")
        
        log_pipeline_step(logger, 3, f"Generating {num_samples} response samples")
        responses = []
        base_temperature = self.llm.temperature
        
        for i in range(num_samples):
            temp = base_temperature - 0.2 + (i * 0.1)
            llm_result = self.llm.generate(prompt, temperature=temp)
            
            if llm_result['status_code'] == 200:
                responses.append({
                    'answer': llm_result['response'],
                    'tokens': llm_result['usage']['total_tokens'],
                    'temperature': temp,
                    'model': llm_result['model']
                })
            else:
                logger.warning(f"Sample {i+1} failed: {llm_result.get('message', 'Unknown error')}")
        
        if not responses:
            log_error(logger, "All self-consistency samples failed")
            return self._handle_llm_error({'status_code': 500, 'message': 'All samples failed'})
        
        log_pipeline_step(logger, 4, "Selecting most consistent response")
        best_response = self._select_most_consistent_response(responses, user_query, analysis)
        
        
        raw_answer = best_response['answer']
        processed_answer = self._embed_source_links(raw_answer, context_docs)
        
        sources = self._extract_sources(context_docs)
        response_time = time.time() - start_time
        
        self.queries_processed += 1
        self.total_response_time += response_time
        
        total_tokens = sum(r['tokens'] for r in responses)
        
        result = {
            "status_code": 200,
            "status": "success",
            "query": user_query,
            "answer": processed_answer,  # Use processed answer
            "sources": sources,
            "metadata": {
                "retrieved_docs": len(context_docs),
                "tokens_used": best_response['tokens'],
                "total_tokens_all_samples": total_tokens,
                "response_time_seconds": round(response_time, 2),
                "model": best_response['model'],
                "filter_type": filter_type,
                "self_consistency_samples": len(responses),
                "method": "self_consistency"
            }
        }
        
        if self.cache_enabled:
            self.cache.set(user_query, result)
        
        log_success(logger, f"Query Complete (Self-Consistency: {response_time:.2f}s, {total_tokens} total tokens)")
        return result

    def _select_most_consistent_response(
        self,
        responses: List[Dict[str, Any]],
        user_query: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the most consistent response from multiple samples."""
        query_part_numbers = set(re.findall(self.PART_NUMBER_PATTERN, user_query, re.IGNORECASE))
        
        scored_responses = []
        
        for i, resp in enumerate(responses):
            answer = resp['answer']
            score = 0
            
            # Check if part numbers from query are mentioned
            answer_part_numbers = set(re.findall(self.PART_NUMBER_PATTERN, answer, re.IGNORECASE))
            if query_part_numbers:
                if answer_part_numbers == query_part_numbers:
                    score += 10
                elif answer_part_numbers.intersection(query_part_numbers):
                    score += 5
                else:
                    score -= 5
            # Check for price mentions
            price_matches = re.findall(r'\$\d+\.?\d*', answer)
            if price_matches:
                score += 3
            # Check for availability mentions
            if re.search(r'\b(in stock|available|out of stock|currently available)\b', answer, re.IGNORECASE):
                score += 2
            # Check for installation video links
            if analysis.get('intent') == 'installation':
                if re.search(r'(install|installation).*video|video.*install', answer, re.IGNORECASE):
                    score += 3
            # Prefer concise responses
            word_count = len(answer.split())
            if 50 <= word_count <= 100:
                score += 2
            elif word_count > 150:
                score -= 1
            elif word_count < 30:
                score -= 2
            # Check for proper formatting
            bold_count = answer.count('**')
            if bold_count >= 2:
                score += 1
            # Prefer responses that don't mention unrelated parts
            if query_part_numbers:
                unrelated_parts = answer_part_numbers - query_part_numbers
                if unrelated_parts:
                    score -= len(unrelated_parts) * 2
            # Check consistency with other responses
            consistency_score = 0
            for other_resp in responses:
                if other_resp == resp:
                    continue
                other_answer = other_resp['answer']
                other_part_numbers = set(re.findall(self.PART_NUMBER_PATTERN, other_answer, re.IGNORECASE))
                
                common_parts = answer_part_numbers.intersection(other_part_numbers)
                if common_parts:
                    consistency_score += len(common_parts)
            
            if consistency_score > 0:
                score += min(consistency_score, 5)
            scored_responses.append((score, resp, i+1))
        
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_resp, best_index = scored_responses[0]
        for score, resp, idx in scored_responses:
            marker = "✓" if idx == best_index else " "
        return best_resp

    def _sanitize_response(self, response_text: str) -> str:
        """
        Sanitize LLM response to remove duplicates, redundant information, and verbatim chunks.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Sanitized response text
        """
        import re
        
        # Step 1: Remove duplicate part numbers in the same sentence/paragraph
        # Pattern: (PS123456) appearing multiple times in close proximity
        part_pattern = self.PART_NUMBER_PATTERN
        parts_found = set()
        
        def deduplicate_parts(match):
            part_id = match.group(0)
            if part_id in parts_found:
                return ""  # Remove duplicate
            parts_found.add(part_id)
            return part_id
        
        # First pass: remove duplicate part numbers in parentheses
        response_text = re.sub(r'\(' + part_pattern + r'\)', deduplicate_parts, response_text)
        
        # Step 2: Remove duplicate blog title citations (e.g., [Title] appearing multiple times)
        
        blog_title_citations_found = set()
        blog_citation_pattern = r'\[([A-Za-z0-9][^\]]{8,})\]'  # Matches [Title] with at least 8 chars (same as conversion)
        
        def deduplicate_blog_citation(match):
            citation_text = match.group(0)  # Full match including brackets
            citation_lower = citation_text.lower()
            if citation_lower in blog_title_citations_found:
                return ""  # Remove duplicate
            blog_title_citations_found.add(citation_lower)
            return citation_text
        
        response_text = re.sub(blog_citation_pattern, deduplicate_blog_citation, response_text)
        
        
        youtube_pattern = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
        response_text = re.sub(youtube_pattern, '', response_text, flags=re.IGNORECASE)
        
        
        redundant_patterns = [
            r'You can find an installation video here:\s*\([^\)]+\)\s*Video\.?',
            r'Installation video:\s*\([^\)]+\)\s*Video\.?',
            r'Video guide:\s*\([^\)]+\)\s*Video\.?',
            r'Watch the video:\s*\([^\)]+\)\s*Video\.?',
            r'follow this video guide[:\s]*\([^\)]*\)',
            r'you can follow this video guide[:\s]*\([^\)]*\)',
            r'installation video[:\s]*\([^\)]*\)',
        ]
        for pattern in redundant_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)
        
        # Step 5: Remove duplicate sentences (if same sentence appears twice)
        sentences = response_text.split('.')
        seen_sentences = set()
        unique_sentences = []
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            # Skip if sentence is too short or already seen
            if len(sentence_clean) < 20:
                unique_sentences.append(sentence)
                continue
            if sentence_clean not in seen_sentences:
                seen_sentences.add(sentence_clean)
                unique_sentences.append(sentence)
        
        response_text = '. '.join(unique_sentences).replace('..', '.')
        
        # Step 6: Fix URLs that have spaces in them (remove spaces from URLs)
        # Pattern: https:// or http:// followed by URL with potential spaces
        # This pattern catches URLs even if they have spaces in the middle
        url_pattern = r'(https?://[^\s\)\]\>]+(?:\s+[^\s\)\]\>]+)*)'
        def fix_url_spaces(match):
            url = match.group(1)
            # Remove spaces from URL
            fixed_url = re.sub(r'\s+', '', url)
            return fixed_url
        
        response_text = re.sub(url_pattern, fix_url_spaces, response_text)
        
        # Step 7: Clean up extra whitespace (but preserve URLs)
        # First, temporarily replace URLs with placeholders
        url_placeholders = {}
        placeholder_counter = 0
        
        def replace_url_with_placeholder(match):
            nonlocal placeholder_counter
            url = match.group(0)
            placeholder = f"__URL_PLACEHOLDER_{placeholder_counter}__"
            url_placeholders[placeholder] = url
            placeholder_counter += 1
            return placeholder
        
        # Replace URLs with placeholders
        url_pattern_for_placeholder = r'https?://[^\s\)]+'
        response_text = re.sub(url_pattern_for_placeholder, replace_url_with_placeholder, response_text)
        
        # Now clean up whitespace
        response_text = re.sub(r'\s+', ' ', response_text)
        response_text = re.sub(r'\s+\.', '.', response_text)
        response_text = response_text.strip()
        
        # Restore URLs
        for placeholder, url in url_placeholders.items():
            response_text = response_text.replace(placeholder, url)
        
        return response_text

    def _titles_match_fuzzy(self, title1: str, title2: str) -> bool:
        """
        Check if two blog titles match using fuzzy logic.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            True if titles match (fuzzy), False otherwise
        """
        import re
        # Normalize both titles
        norm1 = re.sub(r'[^\w\s]', '', title1.lower())
        norm2 = re.sub(r'[^\w\s]', '', title2.lower())
        
        # Check word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # Remove common stop words for better matching
        stop_words = {'a', 'an', 'the', 'to', 'from', 'of', 'in', 'on', 'at', 'for', 'with', 'is', 'are', 'was', 'were'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if len(words1) >= 2 and len(words2) >= 2:  # Reduced minimum from 3 to 2
            common_words = words1.intersection(words2)
            min_words = min(len(words1), len(words2))
            
            return len(common_words) >= max(2, int(min_words * 0.5))
        
        return False
    
    def _embed_source_links(
        self,
        response_text: str,
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Post-process LLM response to replace [Source N] citations with embedded markdown links.
        Also fixes standalone part number patterns and installation video references.
        """
        import re
        
        # Step 0: Fix any URLs that have spaces in them (shouldn't happen, but safety check)
        url_pattern = r'(https?://[^\s\)\]\>]+(?:\s+[^\s\)\]\>]+)*)'
        def fix_url_spaces(match):
            url = match.group(1)
            fixed_url = re.sub(r'\s+', '', url)
            return fixed_url
        response_text = re.sub(url_pattern, fix_url_spaces, response_text)
        
        # Build source mapping: {1: {...urls..., 'type': 'part'}, 2: {...urls..., 'type': 'repair'}, ...}
        source_map = {}
        # Also build part_id -> source mapping for standalone replacement
        part_id_to_source = {}
        
        
        all_context_urls = set()
        
        for idx, doc in enumerate(context_docs, start=1):
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('type', 'unknown')
            
            urls = {'type': doc_type}
            
            if doc_type == 'part':
                if metadata.get('product_url') and metadata.get('product_url') != 'N/A':
                    urls['product'] = metadata['product_url']
                    all_context_urls.add(metadata['product_url'])
                
                # Include installation video URL
                if metadata.get('install_video_url') and metadata.get('install_video_url') != 'N/A':
                    urls['install_video'] = metadata['install_video_url']
                    all_context_urls.add(metadata['install_video_url'])
                
                part_name = metadata.get('part_name', 'Part')
                part_id = metadata.get('part_id', '')
                price = metadata.get('price', '')
                availability = metadata.get('availability', '')
                urls['name'] = part_name
                urls['part_id'] = part_id
                urls['price'] = price
                urls['availability'] = availability
                
                # Map part_id to source number for standalone replacement
                if part_id:
                    part_id_to_source[part_id] = idx
                
            elif doc_type == 'repair':
                
                if metadata.get('detail_url') and metadata.get('detail_url') != 'N/A':
                    urls['detail'] = metadata['detail_url']
                    all_context_urls.add(metadata['detail_url'])
                symptom = metadata.get('symptom', 'Repair Guide')
                urls['name'] = symptom
                
            elif doc_type == 'blog':
                if metadata.get('url') and metadata.get('url') != 'N/A':
                    urls['article'] = metadata['url']
                    all_context_urls.add(metadata['url'])
                title = metadata.get('title', 'Article')
                urls['name'] = title
            
            if urls:
                source_map[idx] = urls
        
        # Step 1: Replace [Source N] citations with embedded links
        citation_pattern = r'\[Source\s+(\d+)\]'
        
        def replace_citation(match):
            source_num = int(match.group(1))
            if source_num not in source_map:
                return match.group(0)
            
            urls = source_map[source_num]
            doc_type = urls.get('type', 'unknown')
            
            if doc_type == 'part':
                part_id = urls.get('part_id', '')
                price = urls.get('price', '')
                availability = urls.get('availability', '')
                
                # Format: (PS123456) on one line, price/stock on next line
                result = f"({part_id})" if part_id else "(Part)"
                
                # Add price/stock on separate line
                if price and price != 'N/A':
                    result += f"\n**${price}**"
                    if availability and availability != 'N/A':
                        result += f" | **{availability}**"
                elif availability and availability != 'N/A':
                    result += f"\n**{availability}**"
                
                # Add links (only PartSelect links, no YouTube videos)
                if 'product' in urls:
                    result += f" [View Part]({urls['product']})"
                
                # Add installation video link if available
                if 'install_video' in urls:
                    result += f" [Installation Video]({urls['install_video']})"
                
                return result
            
            elif doc_type == 'repair':
                source_name = urls.get('name', 'Repair Guide')
                result = f"[{source_name}]({urls['detail']})" if 'detail' in urls else source_name
                
                return result
            
            elif doc_type == 'blog':
                source_name = urls.get('name', 'Article')
                if 'article' in urls:
                    return f"[{source_name}]({urls['article']})"
                return source_name
            
            return match.group(0)
        
        processed_text = re.sub(citation_pattern, replace_citation, response_text)
        
        # Step 2: Convert [Installation video] / [Video guide] citations to links
        # Find parts with installation videos in context
        part_video_map = {}
        for idx, doc in enumerate(context_docs, start=1):
            metadata = doc.get('metadata', {})
            if metadata.get('type') == 'part':
                part_id = metadata.get('part_id', '')
                install_video_url = metadata.get('install_video_url', '')
                if part_id and install_video_url and install_video_url != 'N/A':
                    part_video_map[part_id] = install_video_url
        
        # Convert [Installation video] patterns to links
        installation_video_patterns = [
            r'\[Installation\s+[Vv]ideo\]',
            r'\[Video\s+guide\]',
            r'\[Video\s+Guide\]',
            r'\[Installation\s+video\s+guide\]'
        ]
        
        def replace_installation_video(match):
            # Try to find the part ID from nearby text
            context_start = max(0, match.start() - 100)
            context_end = min(len(processed_text), match.end() + 100)
            context = processed_text[context_start:context_end]
            
            # Extract part number from context
            part_match = re.search(r'\bPS\d+\b', context)
            if part_match:
                part_id = part_match.group(0)
                if part_id in part_video_map:
                    return f"[Installation Video]({part_video_map[part_id]})"
            
            # Fallback: Use first available video from any part in context
            if part_video_map:
                first_video_url = next(iter(part_video_map.values()))
                return f"[Installation Video]({first_video_url})"
            
            # If no video found, remove the citation
            return ''
        
        for pattern in installation_video_patterns:
            processed_text = re.sub(pattern, replace_installation_video, processed_text, flags=re.IGNORECASE)
        
        # Pattern: [Article Title] that matches blog titles from context
        # Build a map of blog titles to URLs (with fuzzy matching support)
        blog_title_map = {}  # Exact match (normalized)
        blog_title_normalized = {}  # Normalized (no punctuation, lowercase)
        
        # First, collect blogs from context_docs
        for idx, doc in enumerate(context_docs, start=1):
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('type', 'unknown')
            if doc_type == 'blog':
                title = metadata.get('title', '')
                url = metadata.get('url', '')
                # Check if URL is valid (not empty, not 'N/A', not None)
                if title and url and url not in ['N/A', '', None]:
                    # Exact match (lowercase)
                    blog_title_map[title.lower()] = {'title': title, 'url': url}
                    # Normalized match (remove punctuation, lowercase)
                    normalized = re.sub(r'[^\w\s]', '', title.lower())
                    blog_title_normalized[normalized] = {'title': title, 'url': url}
        
        # This ensures we're looking at the same text that will be converted
        
        blog_title_pattern_temp = r'\[(["\']?)([A-Za-z0-9][^\]]{3,})\1\]'
        potential_titles_matches = re.findall(blog_title_pattern_temp, processed_text)
        potential_titles = [title.strip('"\'') for quote, title in potential_titles_matches if title]
        
        # Query ChromaDB directly using where filter
        all_blogs_from_store = {}
        try:
            # Access ChromaDB collection directly and query all blogs
            collection = self.pipeline.vectorstore._collection
            
            # SQL-like query: SELECT * FROM collection WHERE type = 'blog'
            blog_results = collection.get(
                where={"type": "blog"},
                limit=1000  # Get all blogs
            )
            
            if blog_results and 'metadatas' in blog_results and blog_results['metadatas']:
                for i, metadata in enumerate(blog_results['metadatas']):
                    if metadata:
                        blog_title = metadata.get('title', '')
                        blog_url = metadata.get('url', '')
                        if blog_title and blog_url and blog_url not in ['N/A', '', None]:
                            blog_title_lower = blog_title.lower()
                            blog_normalized = re.sub(r'[^\w\s]', '', blog_title_lower)
                            all_blogs_from_store[blog_title_lower] = {'title': blog_title, 'url': blog_url}
                            all_blogs_from_store[blog_normalized] = {'title': blog_title, 'url': blog_url}
                            # Also add to the main maps for direct lookup
                            blog_title_map[blog_title_lower] = {'title': blog_title, 'url': blog_url}
                            blog_title_normalized[blog_normalized] = {'title': blog_title, 'url': blog_url}
            else:
                logger.warning("No blogs found in ChromaDB - blog title conversion may fail")
        except Exception as e:
            logger.error(f"Error loading all blogs from ChromaDB: {e}", exc_info=True)
            logger.error("   Blog title conversion may fail due to this error.")
        
        # For each potential title, check if we already have it, if not search vector store
        for title_text in potential_titles:
            title_lower = title_text.lower()
            normalized_title = re.sub(r'[^\w\s]', '', title_lower)
            
            # Skip if we already have this title
            if title_lower in blog_title_map or normalized_title in blog_title_normalized:
                continue
            
            
            if title_lower in all_blogs_from_store:
                blog_info = all_blogs_from_store[title_lower]
                blog_title_map[title_lower] = blog_info
                blog_normalized = re.sub(r'[^\w\s]', '', title_lower)
                blog_title_normalized[blog_normalized] = blog_info
                continue
            
            if normalized_title in all_blogs_from_store:
                blog_info = all_blogs_from_store[normalized_title]
                blog_title_map[title_lower] = blog_info
                blog_title_normalized[normalized_title] = blog_info
                continue
            
            
            found_match = False
            # Get unique blog entries (avoid duplicates from normalized keys)
            unique_blogs = {}
            for key, blog_info in all_blogs_from_store.items():
                # Use title as the unique key
                unique_blogs[blog_info['title'].lower()] = blog_info
            
            for blog_title_key, blog_info in unique_blogs.items():
                if self._titles_match_fuzzy(title_text, blog_info['title']):
                    blog_title_map[title_lower] = blog_info
                    blog_title_normalized[normalized_title] = blog_info
                    found_match = True
                    break
            
            if found_match:
                continue
            
            
            try:
                # Try semantic search first
                blog_result = self.pipeline.search(
                    query=title_text,
                    k=10,  # Increased to get more results
                    filter_metadata={"type": "blog"}
                )
                if blog_result.get('status_code') == 200:
                    blog_docs = blog_result.get('results', [])
                    for blog_doc in blog_docs:
                        blog_metadata = blog_doc.get('metadata', {})
                        blog_title = blog_metadata.get('title', '')
                        blog_url = blog_metadata.get('url', '')
                        if blog_title and blog_url and blog_url not in ['N/A', '', None]:
                            # Check if this blog matches the title we're looking for
                            blog_title_lower = blog_title.lower()
                            blog_normalized = re.sub(r'[^\w\s]', '', blog_title_lower)
                            
                            # Check for match (exact or normalized or fuzzy)
                            if (title_lower == blog_title_lower or 
                                normalized_title == blog_normalized or
                                self._titles_match_fuzzy(title_text, blog_title)):
                                blog_title_map[blog_title_lower] = {'title': blog_title, 'url': blog_url}
                                blog_title_normalized[blog_normalized] = {'title': blog_title, 'url': blog_url}
                                found_match = True
                                break  # Use first match
                
                else:
                    logger.warning(f"  Vector store search failed with status {blog_result.get('status_code')}")
            except Exception as e:
                logger.error(f"Error searching for blog '{title_text}': {e}")
        
        # Pattern: [Title] that matches a blog title
        def convert_blog_title_citation(match):
            # New pattern: group 1 = quote (optional), group 2 = title
            # Old pattern: group 1 = title
            if len(match.groups()) >= 2:
                # New pattern with quote group
                title_in_brackets = match.group(2).strip().strip('"\'')
            else:
                # Old pattern or MatchWrapper
                title_in_brackets = match.group(1).strip().strip('"\'')
            title_lower = title_in_brackets.lower()
            normalized_response = re.sub(r'[^\w\s]', '', title_lower)
            
            if len(blog_title_map) == 0 and len(blog_title_normalized) == 0:
                logger.warning(f"Blog title maps are empty! Cannot convert '{title_in_brackets}'")
                return match.group(0)
            
            if title_lower in blog_title_map:
                blog_info = blog_title_map[title_lower]
                return f"[{blog_info['title']}]({blog_info['url']})"
            
            
            # Check if title_in_brackets is contained in any blog title or vice versa
            best_substring_match = None
            best_overlap = 0
            
            # For very short titles (<= 5 chars), use keyword matching instead of overlap
            if len(title_lower) <= 5:
                # Check if the short title appears as a word in any blog title
                # Normalize words by removing punctuation for comparison
                title_normalized = re.sub(r'[^\w]', '', title_lower)
                for blog_title_lower, blog_info in blog_title_map.items():
                    blog_words_raw = blog_title_lower.split()
                    blog_words_normalized = {re.sub(r'[^\w]', '', word.lower()) for word in blog_words_raw}
                    if title_normalized in blog_words_normalized:
                        # Found as a complete word - this is a strong match
                        return f"[{blog_info['title']}]({blog_info['url']})"
            
            # For longer titles, use overlap-based matching
            for blog_title_lower, blog_info in blog_title_map.items():
                if title_lower in blog_title_lower:
                    overlap = len(title_lower) / len(blog_title_lower)
                    threshold = 0.3 if len(title_lower) <= 6 else 0.6
                    if overlap >= threshold and overlap > best_overlap:
                        best_overlap = overlap
                        best_substring_match = blog_info
                elif blog_title_lower in title_lower:
                    overlap = len(blog_title_lower) / len(title_lower)
                    threshold = 0.3 if len(title_lower) <= 6 else 0.6
                    if overlap >= threshold and overlap > best_overlap:
                        best_overlap = overlap
                        best_substring_match = blog_info
            
            if best_substring_match:
                return f"[{best_substring_match['title']}]({best_substring_match['url']})"
            
            # Try normalized match (remove punctuation)
            if normalized_response in blog_title_normalized:
                blog_info = blog_title_normalized[normalized_response]
                return f"[{blog_info['title']}]({blog_info['url']})"
            
            
            best_norm_substring_match = None
            best_norm_overlap = 0
            for norm_title, blog_info in blog_title_normalized.items():
                # Calculate overlap ratio
                if normalized_response in norm_title:
                    overlap = len(normalized_response) / len(norm_title)
                elif norm_title in normalized_response:
                    overlap = len(norm_title) / len(normalized_response)
                else:
                    continue
                
                threshold = 0.3 if len(normalized_response) <= 6 else 0.6
                if overlap >= threshold and overlap > best_norm_overlap:
                    best_norm_overlap = overlap
                    best_norm_substring_match = blog_info
            
            if best_norm_substring_match:
                return f"[{best_norm_substring_match['title']}]({best_norm_substring_match['url']})"
            
            # Try fuzzy match: check if response title matches any blog title
            best_match = None
            best_score = 0
            for normalized_title, blog_info in blog_title_normalized.items():
                if self._titles_match_fuzzy(title_in_brackets, blog_info['title']):
                    # Calculate match score (number of common words)
                    response_words = set(normalized_response.split())
                    title_words = set(normalized_title.split())
                    common_words = response_words.intersection(title_words)
                    score = len(common_words)
                    if score > best_score:
                        best_score = score
                        best_match = blog_info
            
            if best_match:
                return f"[{best_match['title']}]({best_match['url']})"
            
            
            # Check if the response title contains key words from any blog title
            response_words = set(normalized_response.split())
            if len(response_words) >= 3:  # Only try if we have enough words
                for normalized_title, blog_info in blog_title_normalized.items():
                    blog_words = set(normalized_title.split())
                    # Check if at least 3 key words match (more lenient than fuzzy)
                    common_keywords = response_words.intersection(blog_words)
                    # Focus on important words (exclude common words like "to", "a", "the", "from", "the")
                    important_words = {'how', 'fix', 'repair', 'dishwasher', 'refrigerator', 'leaking', 'leak', 
                                     'not', 'working', 'broken', 'ice', 'maker', 'water', 'filter', 'door', 'seal',
                                     'troubleshooting', 'troubleshoot', 'guide', 'diagnose', 'problem', 'issue'}
                    important_matches = [w for w in common_keywords if w in important_words]
                    
                    if len(common_keywords) >= 3 or len(important_matches) >= 2:
                        return f"[{blog_info['title']}]({blog_info['url']})"
            
            # No match found - try one more time with direct SQL query from vector store as absolute fallback
            
            try:
                # Access ChromaDB collection directly
                collection = self.pipeline.vectorstore._collection
                
                # Query all blogs using where filter (SQL-like)
                blog_results = collection.get(
                    where={"type": "blog"},
                    limit=1000  # Get all blogs
                )
                
                if blog_results and 'metadatas' in blog_results and blog_results['metadatas']:
                    
                    best_match = None
                    best_score = 0
                    
                    for i, metadata in enumerate(blog_results['metadatas']):
                        if metadata:
                            db_title = metadata.get('title', '')
                            db_url = metadata.get('url', '')
                            if db_title and db_url and db_url not in ['N/A', '', None]:
                                db_title_lower = db_title.lower()
                                db_normalized = re.sub(r'[^\w\s]', '', db_title_lower)
                                
                                # Calculate match score
                                score = 0
                                
                                # Exact match (highest score)
                                if title_lower == db_title_lower:
                                    score = 100
                                # Normalized match
                                elif normalized_response == db_normalized:
                                    score = 90
                                # Fuzzy match
                                elif self._titles_match_fuzzy(title_in_brackets, db_title):
                                    # Calculate word overlap score
                                    response_words = set(normalized_response.split())
                                    db_words = set(db_normalized.split())
                                    common_words = response_words.intersection(db_words)
                                    score = len(common_words) * 10  # 10 points per common word
                                # Partial match (check if key words match)
                                else:
                                    response_words = set(normalized_response.split())
                                    db_words = set(db_normalized.split())
                                    common_words = response_words.intersection(db_words)
                                    important_words = {'how', 'fix', 'repair', 'dishwasher', 'refrigerator', 'leaking', 'leak', 
                                                     'not', 'working', 'broken', 'ice', 'maker', 'water', 'filter', 'door', 'seal', 
                                                     'ge', 'troubleshooting', 'troubleshoot', 'guide', 'diagnose', 'problem', 'issue'}
                                    important_matches = [w for w in common_words if w in important_words]
                                    if len(common_words) >= 3 or len(important_matches) >= 2:
                                        score = len(common_words) * 5 + len(important_matches) * 10
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = {'title': db_title, 'url': db_url}
                    
                    if best_match and best_score > 0:
                        blog_info = best_match
                        blog_title_map[title_lower] = blog_info
                        blog_title_normalized[normalized_response] = blog_info
                        return f"[{blog_info['title']}]({blog_info['url']})"
            except Exception as e:
                logger.error(f"Error in SQL query fallback: {e}")
            
            # Extract key words from the title
            response_words = set(normalized_response.split())
            important_keywords = {'how', 'fix', 'repair', 'dishwasher', 'refrigerator', 'leaking', 'leak', 
                                 'not', 'working', 'broken', 'ice', 'maker', 'water', 'filter', 'door', 'seal', 
                                 'ge', 'bottom', 'from', 'top', 'side', 'troubleshooting', 'troubleshoot', 'guide',
                                 'diagnose', 'diagnosis', 'problem', 'issue', 'solution', 'help'}
            title_keywords = [w for w in response_words if w in important_keywords]
            
            if title_keywords and blog_title_normalized:
                best_keyword_match = None
                best_keyword_score = 0
                
                for norm_title, blog_info in blog_title_normalized.items():
                    blog_words = set(norm_title.split())
                    matching_keywords = [kw for kw in title_keywords if kw in blog_words]
                    score = len(matching_keywords)
                    
                    
                    # If we have keywords like "troubleshooting" + "dishwasher" + "leak", even 1-2 matches should work
                    if score >= 1 and score > best_keyword_score:  # At least 1 keyword must match (very lenient)
                        best_keyword_score = score
                        best_keyword_match = blog_info
                
                if best_keyword_match:
                    return f"[{best_keyword_match['title']}]({best_keyword_match['url']})"
            
            # Log available titles for debugging
            if blog_title_normalized:
                # Show similar titles
                response_words = set(normalized_response.split())
                for norm_title, blog_info in list(blog_title_normalized.items())[:5]:
                    title_words = set(norm_title.split())
                    common = response_words.intersection(title_words)
                    if len(common) > 0:
                        pass
            
            if not blog_title_map:
                logger.error("Blog title maps are empty! Vector store may not have blogs loaded.")
            
            return match.group(0)
        
        # Match [Title] patterns that could be blog citations
        # Pattern matches [Title] with capital/lowercase letter or number and at least 4 chars
        # BUT exclude very common/generic patterns like [View Part], [Source N], [Link]
        
        blog_title_pattern = r'\*?\*?\[(["\']?)([A-Za-z0-9][^\]]{3,})\1\]\*?\*?'  # Handles [Title], ["Title"], or **[Title]**
        
        # Pattern to identify non-blog patterns that we should skip
        skip_patterns = {'view part', 'source', 'link', 'more', 'click', 'read', 'see', 'open'}
        
        # Count matches before conversion
        matches_before = len(re.findall(blog_title_pattern, processed_text))
        
        if matches_before > 0:
            found_titles = re.findall(blog_title_pattern, processed_text)
        # Track if any conversions happened
        conversion_count = [0]  # Use list to modify in nested function
        
        def convert_and_count(match):
            original_match = match.group(0)
            # New pattern: group 1 = quote, group 2 = title
            if len(match.groups()) >= 2:
                quote_char = match.group(1) or ''
                title_text = match.group(2).strip().strip('"\'')
            else:
                quote_char = ''
                title_text = match.group(1).strip().strip('"\'')
            title_lower = title_text.lower()
            
            # Skip generic/placeholder patterns
            if title_lower in skip_patterns:
                return original_match
            
            # Create a new match object with cleaned title for conversion
            class MatchWrapper:
                def __init__(self, text, quote=''):
                    self.text = text
                    self.quote = quote
                def group(self, n):
                    if n == 0:
                        return f'[{self.quote}{self.text}{self.quote}]'
                    elif n == 1:
                        return self.quote
                    elif n == 2:
                        return self.text
                    return ''
                def groups(self):
                    return (self.quote, self.text)
            
            wrapped_match = MatchWrapper(title_text, quote_char)
            result = convert_blog_title_citation(wrapped_match)
            
            if '](' in result:
                conversion_count[0] += 1
            return result
        
        # Apply conversion
        processed_text = re.sub(blog_title_pattern, convert_and_count, processed_text)
        
        if conversion_count[0] > 0:
            pass
        else:
            if matches_before > 0:
                def smart_force_convert(match):
                    # Handle both old pattern (group 1 = title) and new pattern (group 1 = quote, group 2 = title)
                    if len(match.groups()) > 1 and match.group(1) in ['"', "'"]:
                        title_text = match.group(2).strip().strip('"\'')
                    else:
                        title_text = match.group(1).strip().strip('"\'')
                    title_lower = title_text.lower()
                    normalized = re.sub(r'[^\w\s]', '', title_lower)
                    title_words = set(normalized.split())
                    
                    # Extract important keywords
                    important_keywords = {'how', 'fix', 'repair', 'dishwasher', 'refrigerator', 'leaking', 'leak', 
                                         'not', 'working', 'broken', 'ice', 'maker', 'water', 'filter', 'door', 'seal',
                                         'troubleshooting', 'troubleshoot', 'guide', 'diagnose', 'problem', 'issue',
                                         'noisy', 'noise', 'making', 'whats', 'what'}
                    title_keywords = [w for w in title_words if w in important_keywords]
                    
                    best_match = None
                    best_score = 0
                    
                    # Try to find the best matching blog based on keywords
                    for blog_title_lower, blog_info in blog_title_map.items():
                        blog_normalized = re.sub(r'[^\w\s]', '', blog_title_lower)
                        blog_words = set(blog_normalized.split())
                        
                        # Score based on keyword matches
                        keyword_matches = len([kw for kw in title_keywords if kw in blog_words])
                        # Also check for word overlap
                        word_overlap = len(title_words.intersection(blog_words))
                        score = keyword_matches * 2 + word_overlap  # Keywords weighted more
                        
                        if score > best_score:
                            best_score = score
                            best_match = blog_info
                    
                    if best_match and best_score >= 2:  # Require at least 2 points (1 keyword or 2 word overlaps)
                        return f"[{best_match['title']}]({best_match['url']})"
                    elif blog_title_map:
                        # Fallback: use first blog if no good match (but log warning)
                        first_blog = next(iter(blog_title_map.values()))
                        return f"[{first_blog['title']}]({first_blog['url']})"
                    else:
                        return match.group(0)  # Return original if no blogs available
                
                processed_text = re.sub(blog_title_pattern, smart_force_convert, processed_text)
        
        markdown_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        markdown_links_found = set()
        
        def deduplicate_markdown_link(match):
            link_text = match.group(0)  # Full markdown link
            link_url = match.group(2).lower()  # URL (normalized to lowercase)
            if link_url in markdown_links_found:
                return ""  # Remove duplicate
            markdown_links_found.add(link_url)
            return link_text
        
        before_dedup = processed_text
        processed_text = re.sub(markdown_link_pattern, deduplicate_markdown_link, processed_text)
        if before_dedup != processed_text:
            pass
        
        # Step 2: Fix standalone part number patterns like "(PS123456, $39.84, in stock)"
        # Pattern: (PS\d+,\s*\$[\d.]+,\s*in stock|available|special order)
        standalone_part_pattern = r'\((' + self.PART_NUMBER_PATTERN + r'),\s*\$([\d.]+),\s*(in stock|available|special order|out of stock|N/A)\)'
        
        def replace_standalone_part(match):
            part_id = match.group(1)
            price = match.group(2)
            availability = match.group(3)
            
            # Find the source for this part_id
            if part_id in part_id_to_source:
                source_num = part_id_to_source[part_id]
                urls = source_map.get(source_num, {})
                
                # Format: (PS123456) on one line, price/stock on next
                result = f"({part_id})\n**${price}** | **{availability}**"
                
                # Add links if available (only PartSelect links, no YouTube videos)
                if 'product' in urls:
                    result += f" [View Part]({urls['product']})"
                
                
                return result
            else:
                # Part not found in sources, just reformat
                return f"({part_id})\n**${price}** | **{availability}**"
        
        processed_text = re.sub(standalone_part_pattern, replace_standalone_part, processed_text, flags=re.IGNORECASE)
        
        
        # Pattern: [View Part] followed by optional text and part number
        view_part_pattern = r'\[View\s+Part\]\s*(?:\((' + self.PART_NUMBER_PATTERN + r')\))?'
        
        def replace_view_part(match):
            part_id = match.group(1) if match.group(1) else None
            
            # Try to find part_id from context if not provided
            if part_id and part_id in part_id_to_source:
                source_num = part_id_to_source[part_id]
                urls = source_map.get(source_num, {})
                if 'product' in urls:
                    return f"[View Part]({urls['product']})"
            
            # If no part_id or not found, try to find any part URL nearby
            # For now, just return the text as-is (will be handled by other patterns)
            return match.group(0)
        
        processed_text = re.sub(view_part_pattern, replace_view_part, processed_text, flags=re.IGNORECASE)
        
        
        
        
        # Pattern: URLs in parentheses like (https://...)
        url_in_parens_pattern = r'\(https?://[^\s\)]+\)'
        
        def remove_duplicate_urls(match):
            url = match.group(0)[1:-1]  # Remove parentheses
            
            if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                return ""  # Remove YouTube URLs
            # Check if this URL is a duplicate PartSelect URL
            if url in all_context_urls:
                return ""  # Remove duplicate URL
            return match.group(0)  # Keep if not a duplicate
        
        processed_text = re.sub(url_in_parens_pattern, remove_duplicate_urls, processed_text)
        
        
        # Pattern: https://... followed by space or end of line
        standalone_url_pattern = r'https?://[^\s\)]+(?=\s|$)'
        
        def remove_standalone_urls(match):
            url = match.group(0)
            
            if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                return ""  # Remove YouTube URLs
            # Check if this URL is a duplicate PartSelect URL
            if url in all_context_urls:
                return ""  # Remove duplicate URL
            return match.group(0)  # Keep if not a duplicate
        
        processed_text = re.sub(standalone_url_pattern, remove_standalone_urls, processed_text)
        
        
        # Note: [Installation video] citations are now converted to links in _embed_source_links
        # No need to remove them here
        
        # Remove "Watch Video" standalone text
        standalone_watch_video = r'\[Watch\s+[Vv]ideo\]'
        processed_text = re.sub(standalone_watch_video, '', processed_text)
        
        # Remove "video guide" text patterns
        video_guide_pattern = r'video\s+guide[:\s]*\([^\)]*\)'
        processed_text = re.sub(video_guide_pattern, '', processed_text, flags=re.IGNORECASE)
        
        # Step 7: Remove any standalone "Sources:" section at the end
        sources_section_pattern = r'\n\n\*\*Sources:\*\*.*$'
        processed_text = re.sub(sources_section_pattern, '', processed_text, flags=re.DOTALL)
        
        sources_section_pattern2 = r'\n\nSources:.*$'
        processed_text = re.sub(sources_section_pattern2, '', processed_text, flags=re.DOTALL)
        
        # Step 8: Clean up extra newlines (max 2 consecutive)
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        
        # Step 9: Clean up any remaining orphaned parentheses or formatting issues
        # Remove parentheses that are now empty: "()" or "( )"
        processed_text = re.sub(r'\(\s*\)', '', processed_text)
        
        return processed_text.strip()