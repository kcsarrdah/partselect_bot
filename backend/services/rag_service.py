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
        enable_retrieval_logging: bool = True  # NEW
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
        
        # NEW: Retrieval quality logging
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
        
        # ✅ FIX: Check cache FIRST (before any processing)
        if self.cache_enabled:
            logger.info("🔍 Checking cache...")
            cached = self.cache.get(user_query)
            if cached:
                self.queries_processed += 1
                response_time = time.time() - start_time
                tokens_saved = cached['metadata'].get('tokens_used', 0)
                log_success(logger, f"Cache hit! (saved {tokens_saved} tokens)")
                cached['metadata']['cache_response_time_seconds'] = round(response_time, 3)
                
                logger.info(f"\n{'='*60}")
                log_success(logger, f"Query Complete from Cache ({response_time:.3f}s)")
                logger.info(f"{'='*60}\n")
                
                return cached
        
        # ✅ CALL: Analyze query
        analysis = self._analyze_query(user_query)
        
        # ✅ CALL: Check if retrieval is needed (NEW)
        if not self._should_retrieve(user_query, analysis):
            logger.info("📝 Query can be answered without retrieval")
            result = self._answer_without_retrieval(user_query, analysis, start_time)
            # ✅ FIX: Cache parametric-only responses too
            if self.cache_enabled:
                self.cache.set(user_query, result)
            return result
        
        # Determine if we should use self-consistency
        is_critical_query = (
            len(analysis.get('part_numbers', [])) > 0 or
            analysis.get('intent') == 'installation'
        )
        
        if use_self_consistency and is_critical_query:
            logger.info("🔄 Critical query detected - using self-consistency")
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
            logger.info(f"\n{'='*60}")
            logger.info(f"RAG Query: {user_query}")
            logger.info(f"{'='*60}")
            
            # Step 1: Retrieve context
            log_pipeline_step(logger, 1, "Retrieving context")
            logger.info(f"🔍 Searching for: '{user_query}' (k={k})")
            context_docs = self._retrieve_context(user_query, k, filter_type)
            
            if not context_docs:
                return self._handle_no_context(user_query)
            
            log_success(logger, f"Retrieved {len(context_docs)} documents")
            
            # Step 2: Build prompt (now includes conflict detection)
            log_pipeline_step(logger, 2, "Building prompt")
            prompt = self._build_prompt(user_query, context_docs, include_examples, analysis)
            log_metric(logger, "Prompt size", f"{len(prompt)} chars")
            
            # Step 3: Generate LLM response
            log_pipeline_step(logger, 3, "Generating response")
            llm_result = self.llm.generate(prompt)
            
            if llm_result['status_code'] != 200:
                return self._handle_llm_error(llm_result)
            
            log_success(logger, "Response generated")
            
            # ✅ NEW: Post-process response to embed source links
            raw_answer = llm_result['response']
            processed_answer = self._embed_source_links(raw_answer, context_docs)
            
            # Step 4: Extract sources
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
                "answer": processed_answer,  # Use processed answer with embedded links
                "sources": sources,  # Keep sources for metadata, but frontend won't display them
                "metadata": {
                    "retrieved_docs": len(context_docs),
                    "tokens_used": llm_result['usage']['total_tokens'],
                    "response_time_seconds": round(response_time, 2),
                    "model": llm_result['model'],
                    "filter_type": filter_type,
                    "method": "standard"
                }
            }
            
            # ✅ CALL: Log retrieval quality (NEW)
            if self.retrieval_logging_enabled:
                self._log_retrieval_quality(
                    query=user_query,
                    retrieved_docs=context_docs,
                    response=llm_result['response'],
                    analysis=analysis
                )
            
            # Cache the result
            if self.cache_enabled:
                self.cache.set(user_query, result)
            
            logger.info(f"\n{'='*60}")
            log_success(logger, f"Query Complete ({response_time:.2f}s, {llm_result['usage']['total_tokens']} tokens)")
            logger.info(f"{'='*60}\n")
            
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
    
    # ✅ NEW METHOD: should_retrieve
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
        
        if any(kw in query_lower for kw in general_keywords):
            # Check if it's appliance-specific (then retrieve)
            appliance_keywords = ['refrigerator', 'dishwasher', 'fridge', 'freezer']
            if not any(app in query_lower for app in appliance_keywords):
                return False
        
        return True  # Default: retrieve
    
    # ✅ NEW METHOD: answer without retrieval
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
        logger.info("📝 Answering from parametric knowledge (no retrieval)")
        
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
    
    # ✅ NEW METHOD: log_retrieval_quality
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
        logger.debug(f"📊 Retrieval Quality: {quality_metrics}")
    
    # ✅ NEW METHOD: check context usage
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
        
        # Extract part numbers
        part_numbers = re.findall(self.PART_NUMBER_PATTERN, query, re.IGNORECASE)
        
        # Extract brand names
        detected_brands = []
        for brand in self.BRAND_KEYWORDS:
            if brand in query_lower:
                detected_brands.append(brand.title())
        
        # Check for repair intent
        is_repair_query = any(keyword in query_lower for keyword in self.REPAIR_KEYWORDS)
        
        # Check for stock intent
        is_stock_query = any(keyword in query_lower for keyword in self.STOCK_KEYWORDS)
        
        # Check for installation intent
        is_installation_query = any(keyword in query_lower for keyword in self.INSTALLATION_KEYWORDS)
        
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
                logger.info("📊 Query complexity: SIMPLE - using direct retrieval")
                return self._simple_retrieval(reformulated_query, k, filter_type, analysis)
            
            elif query_complexity == 'moderate':
                logger.info("📊 Query complexity: MODERATE - using multi-step retrieval")
                return self._moderate_retrieval(reformulated_query, k, filter_type, analysis)
            
            else:  # complex
                logger.info("📊 Query complexity: COMPLEX - using iterative retrieval")
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
            logger.info(f"🔍 Simple retrieval: Looking for exact part match {part_id}")
            
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
                    logger.info(f"✓ Found {len(exact_matches)} exact part match(es)")
                    return exact_matches[:k]
                else:
                    logger.warning(f"⚠️  Part {part_id} not found in vector store")
        
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
            logger.info(f"🔍 Priority search: Looking for exact part match {part_id}")
            
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
                    logger.info(f"✓ Found {len(exact_matches)} exact part match(es)")
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
                else:
                    logger.warning(f"⚠️  Part {part_id} not found in vector store")
        
        # Priority 2: Intent-based search
        if not filter_type:
            if analysis.get('intent') == 'repair':
                logger.info("🔍 Intent: Repair query - retrieving repair guides AND parts")
                
                repair_result = self.pipeline.search_by_type(query, doc_type='repair', k=k//2 + 1)
                if repair_result['status_code'] == 200:
                    repair_results = repair_result.get('results', [])
                    existing_ids = {r.get('metadata', {}).get('doc_id') for r in results}
                    for doc in repair_results:
                        if doc.get('metadata', {}).get('doc_id') not in existing_ids:
                            results.append(doc)
                
                parts_query = query
                if analysis.get('brands'):
                    parts_query = f"{query} {analysis['brands'][0]}"
                
                parts_result = self.pipeline.search_by_type(parts_query, doc_type='part', k=k//2 + 1)
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
                logger.info("🔍 Intent: Stock query - prioritizing parts")
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
                logger.info("🔍 Intent: Installation query - prioritizing installation guides")
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
            logger.info(f"🔍 General search to fill remaining slots ({k - len(results)} needed)")
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
        logger.info(f"✓ Returning {len(final_results)} prioritized results")
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
        
        # Step 1: Initial broad retrieval
        initial_results = self.pipeline.search(query, k=k*2)
        if initial_results['status_code'] == 200:
            results.extend(initial_results.get('results', []))
        
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
        
        # ✅ This calls the UPDATED build_rag_prompt with all new features
        return build_rag_prompt(
            user_query=user_query,
            context_docs=context_docs,
            system_prompt=self.system_prompt,
            include_examples=include_examples,
            query_analysis=analysis
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
        
        # ✅ ADD: Retrieval quality stats
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RAG Query (Self-Consistency): {user_query}")
        logger.info(f"{'='*60}")
        logger.info(f"🔄 Generating {num_samples} samples for consistency check")
        
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
            logger.info(f"  📝 Sample {i+1}/{num_samples} (temperature={temp:.2f})")
            
            llm_result = self.llm.generate(prompt, temperature=temp)
            
            if llm_result['status_code'] == 200:
                responses.append({
                    'answer': llm_result['response'],
                    'tokens': llm_result['usage']['total_tokens'],
                    'temperature': temp,
                    'model': llm_result['model']
                })
                logger.info(f"    ✓ Sample {i+1} generated ({llm_result['usage']['total_tokens']} tokens)")
            else:
                logger.warning(f"    ✗ Sample {i+1} failed: {llm_result.get('message', 'Unknown error')}")
        
        if not responses:
            log_error(logger, "All self-consistency samples failed")
            return self._handle_llm_error({'status_code': 500, 'message': 'All samples failed'})
        
        logger.info(f"✓ Generated {len(responses)}/{num_samples} successful samples")
        
        log_pipeline_step(logger, 4, "Selecting most consistent response")
        best_response = self._select_most_consistent_response(responses, user_query, analysis)
        
        logger.info(f"✓ Selected most consistent response")
        
        # ✅ NEW: Post-process response to embed source links
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
        
        logger.info(f"\n{'='*60}")
        log_success(logger, f"Query Complete (Self-Consistency: {response_time:.2f}s, {total_tokens} total tokens)")
        logger.info(f"{'='*60}\n")
        
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
                    logger.info(f"  Sample {i+1}: Exact part number match (+10)")
                elif answer_part_numbers.intersection(query_part_numbers):
                    score += 5
                    logger.info(f"  Sample {i+1}: Partial part number match (+5)")
                else:
                    score -= 5
                    logger.info(f"  Sample {i+1}: Missing expected part numbers (-5)")
            
            # Check for price mentions
            price_matches = re.findall(r'\$\d+\.?\d*', answer)
            if price_matches:
                score += 3
                logger.info(f"  Sample {i+1}: Contains price information (+3)")
            
            # Check for availability mentions
            if re.search(r'\b(in stock|available|out of stock|currently available)\b', answer, re.IGNORECASE):
                score += 2
                logger.info(f"  Sample {i+1}: Contains availability information (+2)")
            
            # Check for installation video links
            if analysis.get('intent') == 'installation':
                if re.search(r'(install|installation).*video|video.*install', answer, re.IGNORECASE):
                    score += 3
                    logger.info(f"  Sample {i+1}: Mentions installation video (+3)")
            
            # Prefer concise responses
            word_count = len(answer.split())
            if 50 <= word_count <= 100:
                score += 2
                logger.info(f"  Sample {i+1}: Optimal length ({word_count} words, +2)")
            elif word_count > 150:
                score -= 1
                logger.info(f"  Sample {i+1}: Too verbose ({word_count} words, -1)")
            elif word_count < 30:
                score -= 2
                logger.info(f"  Sample {i+1}: Too short ({word_count} words, -2)")
            
            # Check for proper formatting
            bold_count = answer.count('**')
            if bold_count >= 2:
                score += 1
                logger.info(f"  Sample {i+1}: Proper formatting with bold text (+1)")
            
            # Prefer responses that don't mention unrelated parts
            if query_part_numbers:
                unrelated_parts = answer_part_numbers - query_part_numbers
                if unrelated_parts:
                    score -= len(unrelated_parts) * 2
                    logger.info(f"  Sample {i+1}: Mentions {len(unrelated_parts)} unrelated parts (-{len(unrelated_parts) * 2})")
            
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
                logger.info(f"  Sample {i+1}: Consistent with other responses (+{min(consistency_score, 5)})")
            
            scored_responses.append((score, resp, i+1))
        
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_resp, best_index = scored_responses[0]
        logger.info(f"\n📊 Consistency Scores:")
        for score, resp, idx in scored_responses:
            marker = "✓" if idx == best_index else " "
            logger.info(f"  {marker} Sample {idx}: {score} points")
        
        logger.info(f"✓ Selected Sample {best_index} with score {best_score}")
        
        return best_resp

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
        
        # Build source mapping: {1: {...urls..., 'type': 'part'}, 2: {...urls..., 'type': 'repair'}, ...}
        source_map = {}
        # Also build part_id -> source mapping for standalone replacement
        part_id_to_source = {}
        
        for idx, doc in enumerate(context_docs, start=1):
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('type', 'unknown')
            
            urls = {'type': doc_type}
            
            if doc_type == 'part':
                if metadata.get('product_url') and metadata.get('product_url') != 'N/A':
                    urls['product'] = metadata['product_url']
                if metadata.get('install_video_url') and metadata.get('install_video_url') != 'N/A':
                    urls['video'] = metadata['install_video_url']
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
                if metadata.get('video_url') and metadata.get('video_url') != 'N/A':
                    urls['video'] = metadata['video_url']
                if metadata.get('detail_url') and metadata.get('detail_url') != 'N/A':
                    urls['detail'] = metadata['detail_url']
                symptom = metadata.get('symptom', 'Repair Guide')
                urls['name'] = symptom
                
            elif doc_type == 'blog':
                if metadata.get('url') and metadata.get('url') != 'N/A':
                    urls['article'] = metadata['url']
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
                
                # Add links
                if 'product' in urls:
                    result += f" [View Part]({urls['product']})"
                if 'video' in urls:
                    result += f" [Installation Video]({urls['video']})"
                
                return result
            
            elif doc_type == 'repair':
                source_name = urls.get('name', 'Repair Guide')
                result = f"[{source_name}]({urls['detail']})" if 'detail' in urls else source_name
                if 'video' in urls:
                    result += f" [Watch Video]({urls['video']})"
                return result
            
            elif doc_type == 'blog':
                source_name = urls.get('name', 'Article')
                if 'article' in urls:
                    return f"[{source_name}]({urls['article']})"
                return source_name
            
            return match.group(0)
        
        processed_text = re.sub(citation_pattern, replace_citation, response_text)
        
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
                
                # Add links if available
                if 'product' in urls:
                    result += f" [View Part]({urls['product']})"
                if 'video' in urls:
                    result += f" [Installation Video]({urls['video']})"
                
                return result
            else:
                # Part not found in sources, just reformat
                return f"({part_id})\n**${price}** | **{availability}**"
        
        processed_text = re.sub(standalone_part_pattern, replace_standalone_part, processed_text, flags=re.IGNORECASE)
        
        # Step 3: Remove standalone "[Installation video]" or "[Installation Video]" text without links
        # This happens when LLM mentions video but there's no citation
        standalone_video_pattern = r'\[Installation\s+[Vv]ideo\]'
        
        def remove_standalone_video(match):
            # Try to find if there's a video URL in any nearby part
            # For now, just remove it since we can't reliably link it
            return ""  # Remove the text
        
        processed_text = re.sub(standalone_video_pattern, remove_standalone_video, processed_text)
        
        # Also remove "Watch Video" standalone text
        standalone_watch_video = r'\[Watch\s+[Vv]ideo\]'
        processed_text = re.sub(standalone_watch_video, '', processed_text)
        
        # Step 4: Remove any standalone "Sources:" section at the end
        sources_section_pattern = r'\n\n\*\*Sources:\*\*.*$'
        processed_text = re.sub(sources_section_pattern, '', processed_text, flags=re.DOTALL)
        
        sources_section_pattern2 = r'\n\nSources:.*$'
        processed_text = re.sub(sources_section_pattern2, '', processed_text, flags=re.DOTALL)
        
        # Step 5: Clean up extra newlines (max 2 consecutive)
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        
        return processed_text.strip()