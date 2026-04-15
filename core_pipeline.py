"""Utility-only temporal and divergence helpers used by backend api.py."""

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

