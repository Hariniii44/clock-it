"""
Context-Adaptive Evidence Weighting Framework
============================================

Main pipeline showcasing bias-aware evidence weighting algorithm
for Sri Lankan fact-checking with optional Gemini explanations.

This represents the core research contribution: dynamic source weighting
based on political bias alignment with claim direction.
"""
import sys
sys.path.append('src')
from config import Config
from verification.gemini_verification import GeminiClaimVerifier
from datetime import datetime, timedelta
import re

def detect_temporal_claim(claim: str) -> dict:
    """Detect if claim refers to recent events"""
    from datetime import datetime, timedelta
    import calendar
    
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day
    
    temporal_indicators = {
        'immediate': ['today', 'this morning', 'this afternoon', 'this evening', 'tonight', 
                     'just now', 'moments ago', 'minutes ago', 'an hour ago', 'hours ago'],
        'recent': ['yesterday', 'last night', 'this week', 'recent', 'recently', 
                  'latest', 'breaking', 'developing', 'ongoing'],
        'urgent': ['breaking', 'urgent', 'alert', 'emergency', 'crisis'],
        'ongoing_events': ['during', 'amid', 'while', 'as', 'current', 'present', 
                          'operation', 'rescue', 'crisis situation', 'emergency response']
    }
    
    claim_lower = claim.lower()
    temporal_signals = []
    recency_level = 'historical'  # Default
    
    # Generate dynamic month patterns based on current date
    current_month_name = calendar.month_name[current_month].lower()
    
    # Calculate previous month (handle year rollover)
    prev_date = current_date.replace(day=1) - timedelta(days=1)
    prev_month_name = calendar.month_name[prev_date.month].lower()
    prev_year = prev_date.year
    
    # Calculate next month (handle year rollover)
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year
    next_month_name = calendar.month_name[next_month].lower()
    
    # Calculate recent months (last 2-3 months)
    recent_months = []
    for i in range(2, 5):  # 2-4 months ago
        recent_date = current_date.replace(day=1) - timedelta(days=32*i)
        recent_month_name = calendar.month_name[recent_date.month].lower()
        recent_year = recent_date.year
        recent_months.append((recent_month_name, recent_year))
    
    # Dynamic month patterns - PRIORITY CHECK
    month_patterns = [
        # Current month variations
        (rf'{current_month_name}\s+{current_year}', 'current_month'),
        (rf'in\s+{current_month_name}\s+{current_year}', 'current_month'),
        (rf'{current_month_name}\s+of\s+{current_year}', 'current_month'),
        (rf'this\s+{current_month_name}', 'current_month'),
        
        # Previous month
        (rf'{prev_month_name}\s+{prev_year}', 'last_month'),
        (rf'last\s+{prev_month_name}', 'last_month'),
        
        # Next month  
        (rf'{next_month_name}\s+{next_year}', 'next_month'),
        (rf'next\s+{next_month_name}', 'next_month'),
    ]
    
    # Add recent months patterns
    for recent_month_name, recent_year in recent_months:
        month_patterns.append((rf'{recent_month_name}\s+{recent_year}', 'recent_month'))
    
    for pattern, time_ref in month_patterns:
        if re.search(pattern, claim_lower):
            temporal_signals.append(('month_specific', pattern))
            if time_ref == 'current_month':
                recency_level = 'immediate'  # Current month = immediate
                break
            elif time_ref in ['last_month', 'recent_month']:
                recency_level = 'recent'
                break
            elif time_ref == 'next_month':
                recency_level = 'recent'  # Future events are also notable
                break
    
    # Check for general temporal indicators only if no month-specific match
    if recency_level == 'historical':
        for category, indicators in temporal_indicators.items():
            found = [ind for ind in indicators if re.search(r'\b' + re.escape(ind) + r'\b', claim_lower)]
            if found:
                temporal_signals.extend([(category, ind) for ind in found])
                if category in ['immediate', 'urgent']:
                    recency_level = 'immediate'
                elif category == 'recent' and recency_level == 'historical':
                    recency_level = 'recent'
                elif category == 'ongoing_events' and recency_level == 'historical':
                    # Special handling for ongoing events
                    if any(urgent_word in claim_lower for urgent_word in ['rescue', 'operation', 'crisis', 'emergency']):
                        recency_level = 'recent'  # Ongoing operations are likely recent
    
    # Check for relative time expressions
    time_patterns = [
        r'(\d+)\s*(minute|hour|day)s?\s*ago',
        r'this\s+(morning|afternoon|evening|week|month)',
        r'(yesterday|today)',
    ]
    
    for pattern in time_patterns:
        if re.search(pattern, claim_lower):
            temporal_signals.append(('pattern', pattern))
            if 'minute' in pattern or 'hour' in pattern or 'today' in pattern:
                recency_level = 'immediate'
                break
            elif recency_level == 'historical':
                recency_level = 'recent'
    
    # Calculate estimated hours based on analysis
    if recency_level == 'immediate':
        # Check if it's a current month claim
        current_month_mentioned = any(
            time_ref == 'current_month' 
            for pattern, time_ref in month_patterns 
            if re.search(pattern, claim_lower)
        )
        
        if current_month_mentioned:
            # Within current month - calculate days elapsed
            estimated_hours = max(1, (current_day - 1) * 24)  # Days elapsed in current month * 24
        else:
            estimated_hours = 1
    elif recency_level == 'recent':
        estimated_hours = 24  # Default for recent
    else:
        estimated_hours = 168  # Default for historical
    
    return {
        'is_temporal': len(temporal_signals) > 0,
        'recency_level': recency_level,
        'temporal_signals': temporal_signals,
        'estimated_hours_old': estimated_hours,
        'analysis_note': f'Detected as {recency_level} based on current date {current_date.strftime("%Y-%m-%d")}'
    }

def analyze_content_freshness(evidence_list: list, claim: str) -> dict:
    """Analyze evidence content for signs of very recent events, regardless of dates"""
    if not evidence_list:
        return {'freshness_score': 0, 'breaking_indicators': []}
    
    breaking_indicators = []
    freshness_signals = 0
    
    # Look for breaking news language patterns
    breaking_patterns = [
        'breaking', 'just in', 'developing', 'ongoing', 'happening now', 'live updates',
        'update:', 'latest:', 'urgent', 'alert', 'emergency', 'crisis', 'rescue operation',
        'current situation', 'as we speak', 'right now', 'this hour', 'moments ago'
    ]
    
    # Contextual freshness for rescue operations
    rescue_patterns = [
        'rescue operation', 'rescue mission', 'search and rescue', 'emergency response',
        'during operation', 'amid rescue', 'while rescue', 'operation underway',
        'ongoing rescue', 'active rescue', 'current rescue'
    ]
    
    claim_lower = claim.lower()
    is_rescue_claim = any(pattern in claim_lower for pattern in rescue_patterns)
    
    for evidence in evidence_list:
        content = f"{evidence.get('title', '')} {evidence.get('snippet', '')}".lower()
        
        # Count breaking news indicators
        for pattern in breaking_patterns:
            if pattern in content:
                breaking_indicators.append(pattern)
                freshness_signals += 1
        
        # Special weight for rescue operation claims
        if is_rescue_claim:
            for pattern in rescue_patterns:
                if pattern in content:
                    breaking_indicators.append(f"rescue_context: {pattern}")
                    freshness_signals += 2  # Higher weight for contextual match
    
    # Calculate freshness score (0-1)
    max_possible_signals = len(evidence_list) * 3  # Rough estimate
    freshness_score = min(freshness_signals / max_possible_signals, 1.0) if max_possible_signals > 0 else 0
    
    return {
        'freshness_score': freshness_score,
        'breaking_indicators': list(set(breaking_indicators)),  # Remove duplicates
        'total_freshness_signals': freshness_signals,
        'is_rescue_context': is_rescue_claim
    }

def analyze_evidence_temporal_distribution(evidence_list: list) -> dict:
    """Analyze temporal distribution of evidence sources"""
    if not evidence_list:
        return {
            'recent_evidence_ratio': 0,
            'evidence_age_analysis': 'no_evidence',
            'avg_evidence_age_hours': None,
            'oldest_evidence_hours': None,
            'total_sources_with_dates': 0,
            'breaking_news_count': 0,
            'breaking_news_ratio': 0
        }
    
    recent_count = 0
    total_with_dates = 0
    evidence_ages = []
    breaking_news_count = 0  # Count of very fresh sources
    
    current_time = datetime.now()
    
    for evidence in evidence_list:
        date_str = evidence.get('date', '')
        if date_str and date_str != 'Unknown date':
            try:
                # Parse various date formats (ISO + Serper named-month fallback)
                evidence_date = None
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        evidence_date = datetime.strptime(date_str[:10], fmt)
                        break
                    except Exception:
                        continue
                if not evidence_date:
                    for fmt in ['%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y']:
                        try:
                            evidence_date = datetime.strptime(date_str, fmt)
                            break
                        except Exception:
                            continue
                
                if evidence_date:
                    total_with_dates += 1
                    hours_old = (current_time - evidence_date).total_seconds() / 3600
                    evidence_ages.append(hours_old)

                    if hours_old <= 48:  # Within 48 hours
                        recent_count += 1

                    # Count breaking news (less than 12 hours old)
                    if hours_old <= 12:
                        breaking_news_count += 1
            except Exception:
                pass
    
    if total_with_dates == 0:
        return {
            'recent_evidence_ratio': 0,
            'evidence_age_analysis': 'no_dates_available',
            'avg_evidence_age_hours': None,
            'oldest_evidence_hours': None,
            'total_sources_with_dates': 0,
            'breaking_news_count': 0,
            'breaking_news_ratio': 0
        }
    
    recent_ratio = recent_count / len(evidence_list)
    avg_age = sum(evidence_ages) / len(evidence_ages) if evidence_ages else None
    oldest_age = max(evidence_ages) if evidence_ages else None
    
    # Classify evidence age distribution
    if recent_ratio > 0.7:
        age_category = 'mostly_recent'
    elif recent_ratio > 0.3:
        age_category = 'mixed_ages'
    elif avg_age and avg_age > 720:  # Older than 30 days
        age_category = 'mostly_old'
    else:
        age_category = 'moderate_age'
    
    # Calculate breaking news ratio
    breaking_ratio = breaking_news_count / len(evidence_list) if evidence_list else 0
    
    return {
        'recent_evidence_ratio': recent_ratio,
        'evidence_age_analysis': age_category,
        'avg_evidence_age_hours': avg_age,
        'oldest_evidence_hours': oldest_age,
        'total_sources_with_dates': total_with_dates,
        'breaking_news_count': breaking_news_count,
        'breaking_news_ratio': breaking_ratio
    }

def assess_evidence_quality(evidence_list: list, claim_temporal: dict) -> dict:
    """
    Classify evidence sources by type and flag situations where the verdict
    rests mainly on unverified social media rather than established news.

    Three buckets:
      established_news   — proper news domains (hirunews.lk, newswire.lk, …)
      known_news_social  — official social pages of recognised news orgs
      unverified_social  — all other social media accounts/pages
    """
    _SOCIAL_MEDIA_DOMAINS = frozenset({
        'facebook.com', 'x.com', 'twitter.com', 'youtube.com',
        'instagram.com', 'tiktok.com', 'threads.net'
    })

    _KNOWN_HANDLES = {
        'facebook.com': {'adaderana', 'hirunewsenglish', 'hirunews', 'ceylontoday',
                         'dailymirrorlk', 'newsfirstlk', 'themorninglk', 'economynext',
                         'sundaytimesssl'},
        'youtube.com': {'adaderana', 'hirunewsonline'},
        'x.com':       {'adaderana', 'hirunews'},
        'twitter.com': {'adaderana', 'hirunews'},
    }

    def _get_domain(url):
        d = url.split('://', 1)[-1].split('/')[0]
        return d[4:] if d.startswith('www.') else d

    def _get_handle(url):
        try:
            path = url.split('://', 1)[-1].split('/', 1)
            if len(path) < 2:
                return ''
            handle = path[1].split('/')[0].split('?')[0].lstrip('@').lower()
            return '' if handle in ('watch', 'posts', 'videos', 'channel', 'c',
                                    'user', 'reel', 'stories', 'p', '') else handle
        except Exception:
            return ''

    established, known_social, unverified_social = [], [], []

    for ev in evidence_list:
        url = ev.get('link', '')
        domain = _get_domain(url)
        if domain in _SOCIAL_MEDIA_DOMAINS:
            handle = _get_handle(url)
            if handle and handle in _KNOWN_HANDLES.get(domain, set()):
                known_social.append(ev)
            else:
                unverified_social.append(ev)
        else:
            established.append(ev)

    total = len(evidence_list)
    unverified_count = len(unverified_social)
    established_count = len(established)
    unverified_ratio = unverified_count / total if total > 0 else 0
    is_developing = claim_temporal.get('recency_level') in ('immediate', 'recent')

    quality_level = 'good'
    quality_note = None

    if established_count == 0 and total > 0:
        quality_level = 'poor'
        quality_note = (
            f"No established news sources found — all {total} sources are social media posts. "
            f"Treat this verdict with caution and seek corroboration from news outlets."
        )
    elif unverified_ratio > 0.5 and is_developing:
        quality_level = 'poor'
        quality_note = (
            f"Early story: {unverified_count} of {total} sources are unverified social media posts "
            f"with only {established_count} from established outlets. "
            f"Treat as a developing story — the verdict may change as more outlets report."
        )
    elif unverified_ratio > 0.5:
        quality_level = 'moderate'
        quality_note = (
            f"{unverified_count} of {total} sources are social media posts. "
            f"Consider cross-referencing with established news outlets for confirmation."
        )

    return {
        'established_count': established_count,
        'known_news_social_count': len(known_social),
        'unverified_social_count': unverified_count,
        'total': total,
        'unverified_ratio': unverified_ratio,
        'quality_level': quality_level,
        'quality_note': quality_note,
    }


def detect_temporal_mismatch(evidence_list: list) -> dict:
    """
    Detect if evidence spans significantly different time periods.
    Returns details about stale sources so the output can explain the situation.
    """
    current_time = datetime.now()
    dated_sources = []

    for i, evidence in enumerate(evidence_list):
        date_str = evidence.get('date', '')
        if date_str:
            try:
                evidence_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                days_old = (current_time - evidence_date).days
                dated_sources.append({
                    'index': i,
                    'days_old': days_old,
                    'date': date_str,
                    'title': evidence.get('title', '')[:60],
                    'source': evidence.get('source', evidence.get('link', 'unknown'))
                })
            except Exception:
                pass

    if len(dated_sources) < 2:
        return {'has_mismatch': False}

    # Identify the "recent cluster": within 30 days of the newest source
    newest_age = min(s['days_old'] for s in dated_sources)
    recent_threshold = newest_age + 30

    recent_sources = [s for s in dated_sources if s['days_old'] <= recent_threshold]
    stale_sources  = [s for s in dated_sources if s['days_old'] >  recent_threshold]

    if not stale_sources:
        return {'has_mismatch': False}

    # Only flag if the gap between clusters is meaningful (>60 days)
    min_stale_age = min(s['days_old'] for s in stale_sources)
    gap_days = min_stale_age - recent_threshold
    if gap_days < 60:
        return {'has_mismatch': False}

    min_stale_months = max(1, round(min(s['days_old'] for s in stale_sources) / 30))
    max_stale_months = max(1, round(max(s['days_old'] for s in stale_sources) / 30))

    if min_stale_months == max_stale_months:
        age_range = f"~{min_stale_months} month(s)"
    else:
        age_range = f"{min_stale_months}–{max_stale_months} months"

    return {
        'has_mismatch': True,
        'recent_source_count': len(recent_sources),
        'stale_source_count': len(stale_sources),
        'stale_source_indices': [s['index'] for s in stale_sources],
        'gap_days': gap_days,
        'stale_sources': stale_sources,
        'message': (
            f"{len(stale_sources)} source(s) appear to be from {age_range} ago "
            f"and may reflect a different event. They have been down-weighted by the recency penalty."
        )
    }


def calculate_temporal_confidence(claim_temporal: dict, evidence_temporal: dict, total_evidence_count: int) -> dict:
    """
    Assess temporal reliability of the verdict.

    Design principle:
      - Only BREAKING NEWS (immediate claim + scarce evidence) justifies a
        confidence penalty, because the situation itself is genuinely unstable.
      - Everything else is communicated as informational context, not a penalty.
        Historical claims have MORE evidence, not less — penalising them is wrong.
    """
    confidence_multiplier = 1.0
    warnings = []
    is_breaking_news = False

    breaking_count = evidence_temporal.get('breaking_news_count', 0)
    breaking_ratio  = evidence_temporal.get('breaking_news_ratio', 0)
    recency_level   = claim_temporal['recency_level']

    # Breaking news: claim is immediate AND evidence is genuinely scarce
    if recency_level == 'immediate':
        is_breaking_news = True
        if total_evidence_count < 5:
            confidence_multiplier = 0.70
            warnings.append("Very limited evidence for an immediate claim — verdict is PRELIMINARY")
        elif breaking_ratio < 0.3 and total_evidence_count < 8:
            confidence_multiplier = 0.75
            warnings.append("Limited fresh evidence for an immediate claim — verdict may change as story develops")

    # Recent claim with strong freshness signals
    elif recency_level == 'recent' and (breaking_count >= 2 or
            (breaking_count >= 1 and total_evidence_count <= 5)):
        is_breaking_news = True
        confidence_multiplier = 0.85
        warnings.append("Fresh evidence detected — story may still be developing")

    # Claim didn't signal recency but evidence is very fresh — infer breaking news
    elif not claim_temporal['is_temporal'] and breaking_count >= 3:
        is_breaking_news = True
        confidence_multiplier = 0.85
        warnings.append("Evidence suggests this may be a recent event — treat verdict as preliminary")

    # Historical / normal claims: no adjustment
    # (Older claims typically have MORE evidence; penalising them is incorrect.)

    return {
        'temporal_confidence_multiplier': confidence_multiplier,
        'temporal_warnings': warnings,
        'is_breaking_news': is_breaking_news,
        'breaking_evidence_count': breaking_count,
        'should_monitor': is_breaking_news and total_evidence_count < 8,
    }

def _compute_divergence(sources: list) -> str:
    """Classify framing divergence level from ClaimAwareBiasAnalyzer source list."""
    if not sources:
        return 'unknown'
    alignments = [s.get('alignment', 0.0) for s in sources]
    avg = sum(alignments) / len(alignments)
    spread = max(alignments) - min(alignments)
    if avg > 0.6 or spread < 0.2:
        return 'low'
    elif avg > 0.35 or spread < 0.45:
        return 'medium'
    else:
        return 'high'


def _divergence_summary(sources: list, claim_bias: dict) -> str:
    """One-line cross-source divergence description."""
    if not sources:
        return ''
    framings = [s.get('framing_type', 'neutral') for s in sources]
    unique = set(framings)
    claim_framing = claim_bias.get('framing_type', 'neutral')
    if len(unique) == 1:
        return f"All sources use {framings[0]} framing (claim framing: {claim_framing})."
    return (f"Sources vary in framing: {', '.join(sorted(unique))}. "
            f"Claim itself is framed as {claim_framing}.")


def main():
    """Context-adaptive evidence weighting with bias-aware algorithm and temporal awareness"""
    
    print("="*70)
    print("CONTEXT-ADAPTIVE EVIDENCE WEIGHTING FRAMEWORK")
    print("="*70)
    print("Research Focus: Bias-aware dynamic weighting + Temporal awareness for fact-checking")
    
    # Initialize components
    # from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.evidence_retrieval.tavily_evidence_retrieval import TavilyEvidenceRetriever
    from src.evidence_retrieval.google_ai_mode_retrieval import GoogleAIModeRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.verification import ClaimVerifier
    from src.verification.groq_verification import GroqVerifier
    from src.bias_detection import BiasDetector
    from src.bias_detection.framing_analyzer import FramingAnalyzer
    from src.bias_detection.claim_aware_bias_analyzer import ClaimAwareBiasAnalyzer
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator

    # ── Web retriever selection ───────────────────────────────────────────
    # Switch WEB_RETRIEVER here to compare retrieval strategies:
    #   'tavily'        — Tavily API (5 decomposed queries, ~40 results)
    #   'google_ai_mode' — Google AI Mode via SerpAPI (1 query, ~7 curated refs)
    WEB_RETRIEVER = 'google_ai_mode'

    # Core research components
    hybrid_retriever = HybridRetriever()

    if WEB_RETRIEVER == 'google_ai_mode' and Config.SERP_API_KEY:
        web_retriever = GoogleAIModeRetriever(
            serp_api_key=Config.SERP_API_KEY,
            groq_key=Config.GROQ_API_KEY,
            tavily_key=Config.TAVILY_API_KEY,
        )
        print(f"Web retriever: Google AI Mode (SerpAPI)")
    else:
        web_retriever = TavilyEvidenceRetriever(
            tavily_key=Config.TAVILY_API_KEY,
            groq_key=Config.GROQ_API_KEY,
        )
        print(f"Web retriever: Tavily")
    
    # Verification: Groq batch (primary) + DeBERTa per-source (fallback)
    groq_verifier = GroqVerifier(groq_api_key=Config.GROQ_API_KEY)
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    claim_bias_analyzer = ClaimAwareBiasAnalyzer(groq_api_key=Config.GROQ_API_KEY)
    
    # ★ YOUR CORE RESEARCH CONTRIBUTION ★
    weighter = EvidenceWeighter()  # Bias-aware weighting algorithm
    verdict_generator = VerdictGenerator()
    framing_analyzer = FramingAnalyzer(        # kept for fallback / comparison
        groq_key=Config.GROQ_API_KEY,
        bias_profiles=weighter.bias_profiles,
    )
    
    # Optional explanation interface
    try:
        gemini_explainer = GeminiClaimVerifier()
        print("Gemini explanation interface initialized")
        gemini_available = True
    except Exception as e:
        print(f"Gemini explanations unavailable: {e}")
        print("System will run with core algorithm only")
        gemini_available = False
    
    # Get claim to analyze
    print("\n" + "="*70)
    print("Enter claim for bias-aware fact-checking:")
    print("(Add '--skip-gemini' to skip AI explanations and save quota)")
    user_input = input(">> ").strip()
    
    skip_gemini = False
    if "--skip-gemini" in user_input:
        skip_gemini = True
        claim = user_input.replace("--skip-gemini", "").strip()
        print("Gemini explanations disabled to conserve API quota")
    else:
        claim = user_input
    
    if not claim:
        return
    
    print(f"\nAnalyzing: {claim}")
    
    # ==========================================
    # STEP 0: TEMPORAL CLAIM ANALYSIS
    # ==========================================
    print("\n" + "="*70)
    print("STEP 0: TEMPORAL CLAIM ANALYSIS")
    print("="*70)
    
    claim_temporal = detect_temporal_claim(claim)
    
    print(f"Temporal analysis:")
    if claim_temporal['is_temporal']:
        print(f"  WARNING: Temporal claim detected: {claim_temporal['recency_level']} event")
        print(f"  Estimated age: ~{claim_temporal['estimated_hours_old']} hours")
        print(f"  Temporal indicators: {[sig[1] for sig in claim_temporal['temporal_signals'][:3]]}")
        print(f"  Analysis: {claim_temporal.get('analysis_note', 'No additional analysis')}")
        
        if claim_temporal['recency_level'] == 'immediate':
            print(f"  IMMEDIATE EVENT: Limited evidence expected")
        elif claim_temporal['recency_level'] == 'recent':
            print(f"  RECENT EVENT: Evidence may still be emerging")
    else:
        print(f"  Historical claim: Normal evidence availability expected")
    
    # ==========================================
    # STEP 1: DUAL EVIDENCE RETRIEVAL
    # ==========================================
    print("\n" + "="*70)
    print("STEP 1: DUAL EVIDENCE RETRIEVAL")
    print("="*70)
    
    # Database evidence (Government sources)
    print("Retrieving database evidence (government documents)...")
    try:
        database_evidence = hybrid_retriever.hybrid_search(
            query=claim,
            claim_types=None,
            total_results=10,
            use_query_expansion=True
        )
        
        db_formatted = []
        for result in database_evidence:
            formatted = {
                'source': result['source'],
                'title': result['title'],
                'snippet': result.get('passage', result['text']),
                'link': result['url'],
                'dataset_source': result['dataset'],
                'authority': result['authority'],
                'relevance_score': result['similarity_score'],
                'evidence_type': 'database'
            }
            db_formatted.append(formatted)
        
        print(f"  Database sources: {len(db_formatted)}")
        
    except Exception as e:
        print(f"  Database retrieval failed: {e}")
        db_formatted = []
    
    # Web evidence (Independent sources) - SIMPLIFIED RETRIEVAL
    print("Retrieving web evidence (independent sources)...")
    try:
        # Simplified approach: no complex temporal logic, just get most relevant results
        web_evidence = web_retriever.retrieve_hybrid_serper_decomposition(
            claim, 
            num_results=10,
            results_per_query=8
        )
        
        for e in web_evidence:
            e['evidence_type'] = 'web'

        # Display Google AI Mode synthesis if available
        synthesis = next(
            (e.get('google_synthesis') for e in web_evidence if e.get('google_synthesis')),
            None
        )
        if synthesis:
            print("\n" + "─"*70)
            print("GOOGLE AI MODE SYNTHESIS (no bias analysis):")
            print("─"*70)
            for line in synthesis.splitlines():
                if line.strip():
                    print(f"  {line}")
            print("─"*70)

        print(f"\n  Web sources: {len(web_evidence)}")
        
    except Exception as e:
        print(f"  Web retrieval failed: {e}")
        web_evidence = []
    
    # Combine evidence
    all_evidence = db_formatted + web_evidence
    total_sources = len(all_evidence)
    
    print(f"\nTotal evidence collected: {total_sources} sources")
    print(f"   - Government sources: {len(db_formatted)}")
    print(f"   - Independent sources: {len(web_evidence)}")
    
    # ==========================================
    # TEMPORAL EVIDENCE ANALYSIS
    # ==========================================
    evidence_temporal = analyze_evidence_temporal_distribution(all_evidence)
    content_freshness = analyze_content_freshness(all_evidence, claim)
    temporal_confidence = calculate_temporal_confidence(claim_temporal, evidence_temporal, total_sources)
    temporal_mismatch = detect_temporal_mismatch(all_evidence)
    evidence_quality = assess_evidence_quality(all_evidence, claim_temporal)

    print(f"\nEVIDENCE TEMPORAL ANALYSIS:")
    print(f"   Recent evidence ratio: {evidence_temporal['recent_evidence_ratio']:.1%}")
    print(f"   Evidence age profile: {evidence_temporal['evidence_age_analysis']}")

    if evidence_temporal['avg_evidence_age_hours']:
        avg_days = evidence_temporal['avg_evidence_age_hours'] / 24
        print(f"   Average evidence age: {avg_days:.1f} days")

    breaking_count = evidence_temporal.get('breaking_news_count', 0)
    if breaking_count > 0:
        print(f"   Breaking news sources (<12h old): {breaking_count}/{total_sources}")

    if temporal_mismatch['has_mismatch']:
        print(f"   Temporal mismatch detected: {temporal_mismatch['message']}")

    # Source type breakdown
    eq = evidence_quality
    print(f"   Source types: {eq['established_count']} established news, "
          f"{eq['known_news_social_count']} news org social pages, "
          f"{eq['unverified_social_count']} unverified social media")

    if temporal_confidence['temporal_warnings']:
        print(f"\nTEMPORAL WARNINGS:")
        for warning in temporal_confidence['temporal_warnings']:
            print(f"   {warning}")

    if temporal_confidence['should_monitor']:
        print(f"   Recommendation: Monitor for additional evidence over next 2-6 hours")

    # Pre-filter temporally stale sources before NLI when enough fresh sources exist.
    # Stale articles about *different* events (e.g. a 2022 kottu price rise) would
    # otherwise cast NLI votes that conflict with 2026 sources, dragging the verdict
    # toward UNCERTAIN even when fresh evidence clearly supports the claim.
    if temporal_mismatch['has_mismatch'] and temporal_mismatch.get('recent_source_count', 0) >= 3:
        stale_indices = set(temporal_mismatch['stale_source_indices'])
        pre_filter_count = len(stale_indices)
        all_evidence = [e for i, e in enumerate(all_evidence) if i not in stale_indices]
        total_sources = len(all_evidence)
        print(f"\n   Temporal pre-filter: dropped {pre_filter_count} stale source(s) — "
              f"{total_sources} fresh source(s) will be sent to NLI")

    if total_sources == 0:
        print("\nERROR: No evidence found. Cannot proceed with analysis.")
        if claim_temporal['is_temporal']:
            print(f"   Note: This may be due to the recent nature of the claim.")
            print(f"   Suggestion: Try again in a few hours as more sources may publish information.")
        return
    
    # ==========================================
    # STEP 2: NLI VERIFICATION
    # ==========================================
    print("\n" + "="*70)
    print("STEP 2: NLI-BASED VERIFICATION")
    print("="*70)
    
    print("Running NLI verification on all sources...")
    verification_results = []
    bias_analyses = []

    # --- Groq batch verification (1 API call for all sources) ---
    print(f"  Attempting Groq batch verification ({len(all_evidence)} sources)...")
    groq_batch = groq_verifier.verify_batch(claim, all_evidence)
    use_groq = len(groq_batch) == len(all_evidence)
    if use_groq:
        print(f"  Groq batch succeeded — skipping per-source DeBERTa calls")
    else:
        print(f"  Groq batch failed or incomplete — falling back to DeBERTa per source")

    for i, evidence in enumerate(all_evidence, 1):
        try:
            print(f"  Verifying source {i}/{len(all_evidence)}...")

            content_to_analyze = evidence["snippet"]
            print(f"    Source URL: {evidence.get('link', 'No URL')}")
            print(f"    Title: {evidence.get('title', 'No title')[:100]}{'...' if len(evidence.get('title', '')) > 100 else ''}")
            print(f"    Content: {content_to_analyze[:200]}{'...' if len(content_to_analyze) > 200 else ''}")

            if use_groq:
                verification_result = groq_batch[i - 1]
                reason = verification_result.get('reason', '')
                print(f"      Groq verdict: {verification_result['label']} ({verification_result['confidence']:.1%}) — {reason}")
            else:
                verification_result = verifier.verify_claim(claim, content_to_analyze)
                print(f"      RAW MODEL OUTPUT (DeBERTa fallback)")

            if not isinstance(verification_result, dict):
                print(f"    Warning: Unexpected verification result type: {type(verification_result)}")
                continue

            if 'label' not in verification_result or 'confidence' not in verification_result:
                print(f"    Warning: Missing required fields: {list(verification_result.keys())}")
                continue

            verification_results.append(verification_result)

            # Bias analysis (unchanged)
            bias_input = evidence["snippet"]
            print(f"    Running bias analysis...")
            print(f"      Bias input ({len(bias_input)} chars): {bias_input[:300]}{'...' if len(bias_input) > 300 else ''}")
            bias_analysis = bias_detector.analyze_source(bias_input, evidence["link"])
            bias_analyses.append(bias_analysis)

            prof    = bias_analysis.get("source_profile", {})
            combined = bias_analysis.get("combined_score", {})
            print(f"      Bias profile: {prof.get('bias_interpretation','N/A')} "
                  f"(score {prof.get('bias_score',0):+.1f}, conf {prof.get('confidence',0):.1%})")
            print(f"      Text stance: {bias_analysis.get('political_stance',{}).get('stance_score',0):.3f}, "
                  f"emotion {bias_analysis.get('emotional_tone',{}).get('emotion_score',0):.3f}, "
                  f"framing {bias_analysis.get('framing_bias',{}).get('framing_score',0):.3f}")
            print(f"      Combined bias score: {combined.get('overall_bias',0):.3f}")

            status      = verification_result['label']
            confidence  = verification_result['confidence']
            source_type = evidence['evidence_type']
            print(f"  Source {i:2d} [{source_type:8s}]: {status:8s} ({confidence:.1%})")

        except Exception as e:
            print(f"    Error processing source {i}: {e}")
            print(f"    Source type: {evidence.get('evidence_type', 'unknown')}")
            print(f"    Snippet length: {len(evidence.get('snippet', ''))}")
            verification_results.append({'label': 'error', 'confidence': 0.0})
            bias_analyses.append({'overall_bias': 0.0, 'confidence': 0.0})
    
    print(f"\nVerification complete for {total_sources} sources")
    
    # ==========================================
    # STEP 3: ★ BIAS-AWARE EVIDENCE WEIGHTING ★
    # ==========================================
    print("\n" + "="*70)
    print("STEP 3: BIAS-AWARE EVIDENCE WEIGHTING ALGORITHM")
    print("="*70)
    print("CORE RESEARCH CONTRIBUTION: Context-adaptive weighting")
    
    print("\nApplying bias-aware weighting algorithm...")
    print("   Algorithm components:")
    print("   • Source bias profiles (99k Sri Lankan articles)")
    print("   • Dynamic bias-claim alignment calculation") 
    print("   • Authority & recency weighting")
    print("   • Multi-dimensional uncertainty quantification")
    
    # ── Claim-aware bias analysis (single Groq call) ─────────────────────
    # Analyses the claim's own bias then each source relative to that,
    # producing claim-relative alignment scores for the weighting algorithm.
    print("\nRunning claim-aware bias analysis...")
    bias_result = claim_bias_analyzer.analyze(claim, all_evidence)
    claim_bias = bias_result['claim_bias']
    print(f"  Claim framing  : {claim_bias['framing_type']} "
          f"| emotional tone: {claim_bias['emotional_tone']:+.2f} "
          f"| political: {claim_bias['political_direction']}")
    print(f"  Claim bias note: {claim_bias['explanation']}")

    # Build pipeline-compatible framing_result for display + weighter
    framing_result = {
        'sources': [
            {
                'index':                    s['index'],
                'source':                   (all_evidence[s['index']].get('source', '')
                                             if s['index'] < len(all_evidence) else ''),
                'framing_direction':        s['framing_type'],
                'consistency_with_profile': 'unknown',
                'loaded_phrases':           s['loaded_phrases'],
                'explanation':              s['explanation'],
                # extra fields for display
                'emotional_tone':           s['emotional_tone'],
                'political_direction':      s['political_direction'],
                'alignment':                s['alignment'],
            }
            for s in bias_result['sources']
        ],
        'divergence_level':        _compute_divergence(bias_result['sources']),
        'cross_source_divergence': _divergence_summary(bias_result['sources'], claim_bias),
        'contested_entities':      [],
    }
    print(f"  Framing divergence level: {framing_result['divergence_level'].upper()}")

    # Pre-computed alignment scores (claim-relative) for the weighting step
    precomputed_alignments = [s['alignment'] for s in bias_result['sources']]

    # Apply weighting algorithm with claim-relative alignments
    weighted_evidence = weighter.weight_all_evidence(
        claim, all_evidence, verification_results, bias_analyses,
        framing_analyses=framing_result,
        precomputed_alignments=precomputed_alignments,
    )

    print(f"\nWEIGHTING RESULTS:")
    print("─" * 70)

    for i, weighted_item in enumerate(weighted_evidence, 1):
        evidence = weighted_item['evidence']
        weight = weighted_item['weight']
        bias_alignment = weighted_item['bias_alignment']
        verification = weighted_item['verification']
        framing = weighted_item.get('framing_entry', {})

        source_name = evidence.get('source', 'Unknown')[:25]
        source_type = evidence['evidence_type']
        verdict = verification['label']
        framing_dir = framing.get('framing_direction', 'unknown')
        consistency = framing.get('consistency_with_profile', 'unknown')

        print(f"Source {i:2d} | {source_type:8s} | {verdict:8s} | Weight: {weight:.3f} | Bias-align: {bias_alignment:.3f}")
        print(f"         └─ {source_name} | Framing: {framing_dir} | Profile consistency: {consistency}")

    print("\nKey insight: Sources that mirror the CLAIM'S OWN framing are less independent")
    print("   High alignment  → source amplifies claim bias   → bias penalty applied")
    print("   Low alignment   → source frames facts differently → more credible")

    # ── Claim bias summary ────────────────────────────────────────────────
    print(f"\nCLAIM BIAS PROFILE:")
    print("─" * 70)
    print(f"  Emotional tone    : {claim_bias['emotional_tone']:+.2f}  "
          f"(-1=alarming, 0=neutral, +1=positive)")
    print(f"  Political direction: {claim_bias['political_direction']}")
    print(f"  Framing type      : {claim_bias['framing_type']}")
    print(f"  Note              : {claim_bias['explanation']}")

    # ── Per-source bias framing explanations ─────────────────────────────
    framing_sources = framing_result.get('sources', [])
    if framing_sources:
        print(f"\nBIAS FRAMING ANALYSIS (per source):")
        print("─" * 70)
        for entry in framing_sources:
            idx = entry.get('index', 0)
            domain = entry.get('source', 'unknown')
            framing_dir = entry.get('framing_direction', 'unknown')
            emotion = entry.get('emotional_tone', 0.0)
            political = entry.get('political_direction', 'neutral')
            alignment = entry.get('alignment', 0.0)
            phrases = entry.get('loaded_phrases', [])
            explanation = entry.get('explanation', '')

            print(f"  Source {idx + 1:2d} | {domain}")
            print(f"    Framing type      : {framing_dir} | Emotional: {emotion:+.2f} | Political: {political}")
            print(f"    Claim alignment   : {alignment:.2f}  (how much this mirrors the claim's framing)")
            if phrases:
                print(f"    Loaded phrases    : {', '.join(repr(p) for p in phrases[:3])}")
            if explanation:
                print(f"    Explanation       : {explanation}")
            print()

    # ── Cross-source divergence summary ──────────────────────────────────
    divergence = framing_result.get('cross_source_divergence', '')
    contested = framing_result.get('contested_entities', [])
    div_level = framing_result.get('divergence_level', 'unknown')

    print(f"CROSS-SOURCE DIVERGENCE: {div_level.upper()}")
    if divergence:
        print(f"  {divergence}")
    if contested:
        print(f"  Contested entities: {', '.join(contested)}")
    
    # ==========================================
    # STEP 4: VERDICT GENERATION
    # ==========================================
    print("\n" + "="*70)
    print("STEP 4: WEIGHTED VERDICT GENERATION")
    print("="*70)
    
    print("Generating final verdict using weighted evidence...")
    
    # Generate verdict using YOUR algorithm with temporal adjustment
    base_verdict = verdict_generator.generate_verdict(weighted_evidence)
    
    # Apply temporal confidence adjustment
    temporal_adjusted_confidence = base_verdict['confidence'] * temporal_confidence['temporal_confidence_multiplier']
    
    final_verdict = {
        **base_verdict,
        'confidence': temporal_adjusted_confidence,
        'base_confidence': base_verdict['confidence'],
        'temporal_adjustment': temporal_confidence['temporal_confidence_multiplier'],
        'temporal_status': claim_temporal['recency_level'],
        'temporal_warnings': temporal_confidence['temporal_warnings']
    }
    
    verdict_label = final_verdict['verdict']
    if temporal_confidence['is_breaking_news']:
        verdict_label += " (PRELIMINARY)"

    print(f"\nFINAL VERDICT:")
    print("─" * 40)
    print(f"Verdict: {verdict_label}")
    print(f"Confidence: {final_verdict['confidence']:.1%}")

    # Only explain the temporal adjustment when it was actually applied (breaking news)
    if temporal_confidence['is_breaking_news'] and temporal_confidence['temporal_confidence_multiplier'] < 1.0:
        print(f"  └─ Base algorithmic confidence: {final_verdict['base_confidence']:.1%}")
        print(f"  └─ Breaking news adjustment: ×{temporal_confidence['temporal_confidence_multiplier']:.2f}")
        print(f"  └─ Reason: {total_sources} sources available for a developing story")
    print(f"")
    print(f"Vote Distribution:")
    print(f"  Support:  {final_verdict['support_score']:.1%}")
    print(f"  Refute:   {final_verdict['refute_score']:.1%}")
    print(f"  Neutral:  {final_verdict['neutral_score']:.1%}")
    
    # Show uncertainty breakdown
    uncertainty = final_verdict.get('uncertainty_decomposition', {})
    if uncertainty:
        print(f"\nUncertainty Analysis:")
        print(f"  Epistemic (model variance): {uncertainty.get('epistemic', 0):.3f}")
        print(f"  Aleatoric (insufficient evidence): {uncertainty.get('aleatoric', 0):.3f}")
        print(f"  Bias-induced (source conflicts): {uncertainty.get('bias_induced', 0):.3f}")
    
    print(f"\nFactual claim detected: {final_verdict.get('is_factual_claim', False)}")
    print(f"Source diversity score: {final_verdict.get('diversity_score', 0):.3f}")
    
    # ==========================================
    # TEMPORAL STATUS SUMMARY
    # ==========================================
    show_summary = (
        temporal_confidence['is_breaking_news'] or
        temporal_mismatch['has_mismatch'] or
        bool(temporal_confidence['temporal_warnings']) or
        evidence_quality['quality_level'] != 'good'
    )

    if show_summary:
        print(f"\nTEMPORAL STATUS SUMMARY:")
        print("─" * 50)

        if temporal_confidence['is_breaking_news']:
            print(f"Status: BREAKING / DEVELOPING STORY")
            print(f"   • Confidence has been reduced to reflect limited evidence")
            print(f"   • This verdict is PRELIMINARY — information may change rapidly")
            if temporal_confidence['should_monitor']:
                print(f"   • Check back in 2-6 hours for updates")

        if temporal_mismatch['has_mismatch']:
            print(f"\nTEMPORAL MISMATCH NOTE:")
            print(f"   • {temporal_mismatch['message']}")
            print(f"   • Confidence score reflects only the most recent sources")
            if temporal_mismatch['stale_sources']:
                print(f"   • Stale sources:")
                for s in temporal_mismatch['stale_sources'][:3]:
                    print(f"     - {s['title']} (~{s['days_old']} days old)")

        if evidence_quality['quality_note']:
            level_prefix = "WARNING: " if evidence_quality['quality_level'] == 'poor' else "NOTE: "
            print(f"\nEVIDENCE QUALITY NOTE:")
            print(f"   {level_prefix}{evidence_quality['quality_note']}")
    
    # ==========================================
    # STEP 5: OPTIONAL GEMINI EXPLANATIONS
    # ==========================================
    if gemini_available and not skip_gemini:
        print("\n" + "="*70)
        print("STEP 5: NATURAL LANGUAGE EXPLANATIONS (Optional Enhancement)")
        print("="*70)
        print("Generating user-friendly explanations with Gemini...")
        
        try:
            # Explain weighting decisions with temporal context
            print("\nWEIGHTING DECISION EXPLANATIONS:")
            print("─" * 50)
            
            # Create temporal context for Gemini
            temporal_context = {
                'is_breaking_news': temporal_confidence.get('is_breaking_news', False),
                'recency_level': claim_temporal['recency_level'],
                'estimated_hours_old': claim_temporal['estimated_hours_old'],
                'content_freshness_score': content_freshness['freshness_score'],
                'breaking_indicators': content_freshness.get('breaking_indicators', []),
                'temporal_warnings': temporal_confidence['temporal_warnings']
            }
            
            weighting_explanations = gemini_explainer.explain_weighting_decisions(
                claim, weighted_evidence, final_verdict, temporal_context
            )
            
            print(weighting_explanations)
            
            # Explain bias patterns with temporal context
            print(f"\nBIAS PATTERN EXPLANATIONS:")
            print("─" * 50)
            
            bias_explanation = gemini_explainer.explain_bias_patterns(
                claim, all_evidence, bias_analyses, temporal_context
            )
            
            # Show key insights from bias analysis
            if hasattr(bias_explanation, 'specific_examples') and bias_explanation.specific_examples:
                print("Cross-source framing examples:")
                for example in bias_explanation.specific_examples[:2]:
                    print(f"  • {example.get('comparison_type', 'Comparison')}")
                    print(f"    {example.get('source_a', '')}")
                    print(f"    {example.get('source_b', '')}")
                    print(f"    Analysis: {example.get('bias_difference', '')}")
                    print()
            
            # User recommendations
            if hasattr(bias_explanation, 'recommendations') and bias_explanation.recommendations:
                print("User recommendations:")
                for i, rec in enumerate(bias_explanation.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
        except Exception as e:
            print(f"Gemini explanations failed: {e}")
            print("Core algorithm results remain valid")
    
    # ==========================================
    # SYSTEM SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("SYSTEM PERFORMANCE SUMMARY")
    print("="*70)
    
    print(f"Temporal Analysis: {claim_temporal['recency_level']} claim detected")
    print(f"Evidence Retrieval: {total_sources} sources from dual architecture")
    print(f"Evidence Temporal Profile: {evidence_temporal['evidence_age_analysis']}")
    print(f"NLI Verification: Consistent single-method approach")
    print(f"Bias-Aware Weighting: Dynamic weights applied to {total_sources} sources")
    print(f"Verdict Generation: {final_verdict['verdict']} at {final_verdict['confidence']:.1%} confidence")
    
    if temporal_confidence['temporal_confidence_multiplier'] < 1.0:
        print(f"Temporal Confidence Adjustment: Applied {temporal_confidence['temporal_confidence_multiplier']:.1f}× multiplier")
    
    if gemini_available and not skip_gemini:
        print(f"User Explanations: Natural language interface active")
    elif skip_gemini:
        print(f"User Explanations: Skipped to conserve API quota")
    else:
        print(f"User Explanations: Core algorithm only (Gemini unavailable)")
    
    print(f"\nRESEARCH CONTRIBUTION DEMONSTRATED:")
    print(f"   • Context-adaptive evidence weighting framework")
    print(f"   • Political bias-claim alignment algorithm")  
    print(f"   • Multi-dimensional uncertainty quantification")
    print(f"   • Sri Lankan media bias profile integration")
    print(f"   • Temporal evidence availability assessment")
    print(f"   • Real-time claim detection and confidence adjustment")
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Show quota management info if Gemini was used
    if gemini_available and not skip_gemini:
        print("\nGEMINI API USAGE:")
        print("─" * 30)
        print("This run used 2 API calls (weighting + bias explanations)")
        print("Free tier limit: ~2 requests/minute, ~20 requests/day")
        print("To conserve quota: Use '--skip-gemini' flag in future runs")
        print("To increase limits: Set up billing at console.cloud.google.com")

if __name__ == "__main__":
    main()