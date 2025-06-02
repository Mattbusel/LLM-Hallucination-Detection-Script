import re
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import hashlib


@dataclass
class DetectionResult:
    """Result of hallucination detection analysis"""
    hallucination_probability: float
    confidence_score: float
    detected_issues: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]


class HallucinationDetector:
    """
    Multi-method hallucination detection for LLM responses
    """
    
    def __init__(self):
        # Common hallucination patterns
        self.uncertainty_phrases = [
            "i think", "i believe", "might be", "could be", "possibly", "perhaps",
            "it seems", "appears to", "likely", "probably", "may have", "seems like",
            "i'm not sure", "uncertain", "unclear", "not certain"
        ]
        
        self.overconfidence_phrases = [
            "definitely", "certainly", "absolutely", "without doubt", "guaranteed",
            "always", "never", "impossible", "undoubtedly", "unquestionably",
            "100%", "completely certain", "no question"
        ]
        
        self.hedging_phrases = [
            "according to", "based on", "it is said", "reportedly", "allegedly",
            "supposedly", "claimed", "purported", "rumored"
        ]
        
        # Factual claim indicators
        self.factual_indicators = [
            r'\d{4}',  # Years
            r'\$[\d,]+',  # Money amounts
            r'\d+%',  # Percentages
            r'\d+\.?\d*\s*(million|billion|thousand)',  # Large numbers
            r'(founded|established|created|born|died)\s+(in\s+)?\d{4}',  # Dates
        ]
    
    def analyze_response(
        self, 
        response: str, 
        context: Optional[str] = None,
        confidence_threshold: float = 0.7,
        enable_consistency_check: bool = True
    ) -> DetectionResult:
        """
        Main analysis method - returns comprehensive hallucination assessment
        """
        issues = []
        metrics = {}
        recommendations = []
        
        # 1. Confidence Analysis
        confidence_score = self._analyze_confidence_patterns(response)
        metrics['confidence_inconsistency'] = confidence_score
        
        if confidence_score > 0.6:
            issues.append("High confidence inconsistency detected")
            recommendations.append("Review factual claims for accuracy")
        
        # 2. Factual Density Analysis
        factual_density = self._calculate_factual_density(response)
        metrics['factual_density'] = factual_density
        
        if factual_density > 0.3:
            issues.append("High density of specific factual claims")
            recommendations.append("Verify numerical data and specific facts")
        
        # 3. Coherence Analysis
        coherence_score = self._analyze_coherence(response)
        metrics['coherence_score'] = coherence_score
        
        if coherence_score < 0.5:
            issues.append("Low coherence in response structure")
            recommendations.append("Check for logical inconsistencies")
        
        # 4. Context Consistency (if context provided)
        if context:
            context_consistency = self._check_context_consistency(response, context)
            metrics['context_consistency'] = context_consistency
            
            if context_consistency < 0.6:
                issues.append("Response inconsistent with provided context")
                recommendations.append("Ensure response aligns with given information")
        
        # 5. Repetition and Contradiction Detection
        repetition_score = self._detect_repetitions(response)
        contradiction_score = self._detect_contradictions(response)
        
        metrics['repetition_score'] = repetition_score
        metrics['contradiction_score'] = contradiction_score
        
        if repetition_score > 0.4:
            issues.append("Excessive repetition detected")
        
        if contradiction_score > 0.3:
            issues.append("Internal contradictions found")
            recommendations.append("Review response for conflicting statements")
        
        # 6. Calculate overall hallucination probability
        hallucination_prob = self._calculate_hallucination_probability(metrics)
        
        # 7. Generate final recommendations
        if hallucination_prob > confidence_threshold:
            recommendations.extend([
                "Consider regenerating response",
                "Fact-check specific claims",
                "Request sources or citations"
            ])
        
        return DetectionResult(
            hallucination_probability=hallucination_prob,
            confidence_score=1 - hallucination_prob,
            detected_issues=issues,
            metrics=metrics,
            recommendations=list(set(recommendations))  # Remove duplicates
        )
    
    def _analyze_confidence_patterns(self, text: str) -> float:
        """Analyze confidence/uncertainty language patterns"""
        text_lower = text.lower()
        
        uncertainty_count = sum(1 for phrase in self.uncertainty_phrases if phrase in text_lower)
        overconfidence_count = sum(1 for phrase in self.overconfidence_phrases if phrase in text_lower)
        hedging_count = sum(1 for phrase in self.hedging_phrases if phrase in text_lower)
        
        # Normalize by text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        # High uncertainty or overconfidence both indicate potential issues
        uncertainty_ratio = uncertainty_count / word_count * 100
        overconfidence_ratio = overconfidence_count / word_count * 100
        hedging_ratio = hedging_count / word_count * 100
        
        # Combine scores (uncertainty and overconfidence are both red flags)
        confidence_inconsistency = min(1.0, (uncertainty_ratio + overconfidence_ratio * 1.5 - hedging_ratio * 0.5) / 2)
        
        return max(0.0, confidence_inconsistency)
    
    def _calculate_factual_density(self, text: str) -> float:
        """Calculate density of specific factual claims"""
        factual_matches = 0
        
        for pattern in self.factual_indicators:
            factual_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        return min(1.0, factual_matches / word_count * 10)
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze logical coherence of the response"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence metrics
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_variance = sum((len(s.split()) - avg_sentence_length) ** 2 for s in sentences) / len(sentences)
        
        # Normalize variance (lower variance = better coherence)
        coherence = max(0.0, 1.0 - min(1.0, length_variance / 100))
        
        return coherence
    
    def _check_context_consistency(self, response: str, context: str) -> float:
        """Check if response is consistent with provided context"""
        # Simple word overlap method
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        if not context_words:
            return 1.0
        
        overlap = len(response_words & context_words)
        consistency = overlap / len(context_words)
        
        return min(1.0, consistency * 2)  # Scale up for better sensitivity
    
    def _detect_repetitions(self, text: str) -> float:
        """Detect excessive repetition in response"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Check for repeated sentences
        sentence_counts = Counter(sentences)
        repeated_sentences = sum(count - 1 for count in sentence_counts.values() if count > 1)
        
        repetition_ratio = repeated_sentences / len(sentences)
        return min(1.0, repetition_ratio * 2)
    
    def _detect_contradictions(self, text: str) -> float:
        """Simple contradiction detection"""
        sentences = re.split(r'[.!?]+', text.lower())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Look for contradictory patterns
        contradiction_patterns = [
            (r'\bnot\b', r'\bis\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
        ]
        
        contradictions = 0
        for sentence in sentences:
            for neg_pattern, pos_pattern in contradiction_patterns:
                if re.search(neg_pattern, sentence) and re.search(pos_pattern, sentence):
                    contradictions += 1
        
        return min(1.0, contradictions / len(sentences) * 5)
    
    def _calculate_hallucination_probability(self, metrics: Dict[str, float]) -> float:
        """Calculate overall hallucination probability from all metrics"""
        # Weighted combination of all metrics
        weights = {
            'confidence_inconsistency': 0.25,
            'factual_density': 0.20,
            'coherence_score': -0.15,  # Negative because higher coherence = lower hallucination
            'context_consistency': -0.20,  # Negative because higher consistency = lower hallucination
            'repetition_score': 0.15,
            'contradiction_score': 0.35
        }
        
        probability = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                probability += value * weight
                total_weight += abs(weight)
        
        # Normalize to 0-1 range
        if total_weight > 0:
            probability = probability / total_weight
        
        return max(0.0, min(1.0, probability + 0.3))  # Add baseline probability


# Convenience functions for easy integration
def quick_hallucination_check(response: str, threshold: float = 0.7) -> bool:
    """Quick boolean check for hallucination"""
    detector = HallucinationDetector()
    result = detector.analyze_response(response, confidence_threshold=threshold)
    return result.hallucination_probability > threshold


def get_hallucination_score(response: str, context: str = None) -> float:
    """Get just the hallucination probability score"""
    detector = HallucinationDetector()
    result = detector.analyze_response(response, context=context)
    return result.hallucination_probability


def analyze_with_recommendations(response: str, context: str = None) -> Dict[str, Any]:
    """Full analysis with actionable recommendations"""
    detector = HallucinationDetector()
    result = detector.analyze_response(response, context=context)
    
    return {
        'hallucination_probability': result.hallucination_probability,
        'confidence': result.confidence_score,
        'issues': result.detected_issues,
        'recommendations': result.recommendations,
        'detailed_metrics': result.metrics
    }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: High confidence claims
    test_response_1 = """
    The Eiffel Tower was definitely built in 1887 and is exactly 324 meters tall. 
    It was designed by Gustave Eiffel and cost exactly $1.2 million to construct. 
    Without doubt, it receives 7 million visitors every year.
    """
    
    # Example 2: Uncertain response
    test_response_2 = """
    I think the Eiffel Tower might have been built around the late 1800s. 
    It seems to be quite tall, possibly around 300 meters or so. 
    I believe it was designed by someone named Eiffel, but I'm not entirely sure about the details.
    """
    
    # Example 3: Contradictory response
    test_response_3 = """
    Python is always the best programming language for data science. 
    However, Python is never suitable for machine learning projects. 
    It's impossible to use Python for AI, but it's definitely the top choice for AI development.
    """
    
    detector = HallucinationDetector()
    
    print("=== Hallucination Detection Results ===\n")
    
    for i, response in enumerate([test_response_1, test_response_2, test_response_3], 1):
        print(f"Test Response {i}:")
        result = detector.analyze_response(response)
        
        print(f"Hallucination Probability: {result.hallucination_probability:.2f}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Issues Found: {', '.join(result.detected_issues) if result.detected_issues else 'None'}")
        print(f"Recommendations: {', '.join(result.recommendations) if result.recommendations else 'None'}")
        print("-" * 50)
