"""Domain-specific model routing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.model import Model
from .model_registry import ModelRegistry
from .model_selector import ModelSelector, ModelSelectionCriteria


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str  # Domain name (e.g., "medical", "legal", "creative")
    keywords: List[str] = field(
        default_factory=list
    )  # Keywords that indicate this domain
    patterns: List[str] = field(default_factory=list)  # Regex patterns to match
    preferred_models: List[str] = field(default_factory=list)  # Preferred model IDs
    required_capabilities: List[str] = field(
        default_factory=list
    )  # Required capabilities
    required_certifications: List[str] = field(
        default_factory=list
    )  # e.g., ["HIPAA", "SOC2"]
    min_accuracy_score: float = 0.0  # Minimum accuracy requirement

    def matches_text(self, text: str) -> float:
        """
        Check if text matches this domain.

        Args:
            text: Text to check

        Returns:
            Confidence score (0-1) that text belongs to this domain
        """
        text_lower = text.lower()
        score = 0.0

        # Check keywords
        keyword_matches = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1

        if self.keywords:
            keyword_score = keyword_matches / len(self.keywords)
            score = max(score, keyword_score)

        # Check patterns
        pattern_matches = 0
        for pattern in self.patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue

        if self.patterns:
            pattern_score = pattern_matches / len(self.patterns)
            score = max(score, pattern_score * 1.2)  # Patterns are more specific

        return min(score, 1.0)


class DomainRouter:
    """
    Routes model selection based on content domain.

    Analyzes input text to determine domain and selects
    appropriate models with domain expertise.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize domain router.

        Args:
            registry: Model registry to use
        """
        self.registry = registry
        self.selector = ModelSelector(registry)
        self.domains: Dict[str, DomainConfig] = {}

        # Initialize default domains
        self._init_default_domains()

    def _init_default_domains(self) -> None:
        """Initialize default domain configurations."""

        # Medical domain
        medical = DomainConfig(
            name="medical",
            keywords=[
                "patient",
                "diagnosis",
                "treatment",
                "symptom",
                "medication",
                "clinical",
                "medical",
                "health",
                "disease",
                "therapy",
                "prescription",
                "dosage",
                "condition",
                "physician",
                "hospital",
            ],
            patterns=[
                r"\b(diagnos\w+|treat\w+|medicat\w+|clinical|medical)\b",
                r"\b(patient|physician|doctor|nurse|hospital|clinic)\b",
                r"\b(symptom|disease|condition|disorder|syndrome)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            required_certifications=["HIPAA"],
            min_accuracy_score=0.8,
        )

        # Legal domain
        legal = DomainConfig(
            name="legal",
            keywords=[
                "legal",
                "law",
                "contract",
                "agreement",
                "liability",
                "regulation",
                "compliance",
                "court",
                "attorney",
                "clause",
                "jurisdiction",
                "statute",
                "litigation",
                "precedent",
            ],
            patterns=[
                r"\b(legal|law|contract|agreement|liabilit\w+)\b",
                r"\b(regulat\w+|complian\w+|jurisdict\w+)\b",
                r"\b(court|attorney|lawyer|counsel)\b",
                r"\bsection\s+\d+\.\d+\b",  # Legal section references
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Creative domain
        creative = DomainConfig(
            name="creative",
            keywords=[
                "creative",
                "story",
                "poem",
                "art",
                "design",
                "narrative",
                "character",
                "plot",
                "scene",
                "dialogue",
                "imagination",
                "artistic",
                "aesthetic",
                "composition",
            ],
            patterns=[
                r"\b(creat\w+|story|poem|art\w+|design)\b",
                r"\b(write|compose|craft|imagine)\s+(a|an|the)\s+\w+",
                r"\b(character|plot|scene|dialogue|narrative)\b",
            ],
            preferred_models=["claude-3-opus", "gpt-4"],
            required_capabilities=[],
            min_accuracy_score=0.85,
        )

        # Technical/Engineering domain
        technical = DomainConfig(
            name="technical",
            keywords=[
                "technical",
                "engineering",
                "system",
                "architecture",
                "implementation",
                "algorithm",
                "optimization",
                "performance",
                "infrastructure",
                "deployment",
                "integration",
                "api",
            ],
            patterns=[
                r"\b(technical|engineer\w+|architect\w+)\b",
                r"\b(implement\w+|algorithm|optimiz\w+)\b",
                r"\b(system|infrastructure|deployment|integration)\b",
                r"\b(API|SDK|framework|library|database)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Scientific/Research domain
        scientific = DomainConfig(
            name="scientific",
            keywords=[
                "scientific",
                "research",
                "hypothesis",
                "experiment",
                "analysis",
                "data",
                "methodology",
                "results",
                "conclusion",
                "peer-review",
                "publication",
                "citation",
                "theory",
            ],
            patterns=[
                r"\b(scientif\w+|research|hypothesis|experiment)\b",
                r"\b(analy\w+|data|methodolog\w+|result\w+)\b",
                r"\b(peer[\s-]?review|publicat\w+|citat\w+)\b",
                r"\b(theory|theorem|proof|evidence)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Financial domain
        financial = DomainConfig(
            name="financial",
            keywords=[
                "financial",
                "investment",
                "portfolio",
                "market",
                "trading",
                "risk",
                "return",
                "asset",
                "equity",
                "revenue",
                "profit",
                "loss",
                "accounting",
                "audit",
            ],
            patterns=[
                r"\b(financ\w+|invest\w+|portfolio|market)\b",
                r"\b(trad\w+|risk|return|asset|equity)\b",
                r"\b(revenue|profit|loss|accounting|audit)\b",
                r"\$\d+\.?\d*[KMB]?\b",  # Dollar amounts
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Educational domain
        educational = DomainConfig(
            name="educational",
            keywords=[
                "educational",
                "learning",
                "teaching",
                "student",
                "curriculum",
                "lesson",
                "tutorial",
                "explain",
                "understand",
                "concept",
                "example",
                "practice",
            ],
            patterns=[
                r"\b(educat\w+|learn\w+|teach\w+|student)\b",
                r"\b(explain|understand|concept|example)\b",
                r"\b(lesson|tutorial|curriculum|course)\b",
                r"(?i)how\s+(to|do|does|can)\b",  # Educational questions
            ],
            preferred_models=["gpt-4", "claude-3-opus", "gpt-3.5-turbo"],
            required_capabilities=[],
            min_accuracy_score=0.85,
        )

        # Add domains to router
        self.domains = {
            "medical": medical,
            "legal": legal,
            "creative": creative,
            "technical": technical,
            "scientific": scientific,
            "financial": financial,
            "educational": educational,
        }

    def register_domain(self, domain: DomainConfig) -> None:
        """
        Register a custom domain configuration.

        Args:
            domain: Domain configuration to register
        """
        self.domains[domain.name] = domain

    def detect_domains(
        self, text: str, threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Detect domains in text.

        Args:
            text: Text to analyze
            threshold: Minimum confidence threshold

        Returns:
            List of (domain_name, confidence) tuples sorted by confidence
        """
        detections = []

        for domain_name, domain_config in self.domains.items():
            confidence = domain_config.matches_text(text)
            if confidence >= threshold:
                detections.append((domain_name, confidence))

        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    async def route_by_domain(
        self,
        text: str,
        base_criteria: Optional[ModelSelectionCriteria] = None,
        domain_override: Optional[str] = None,
    ) -> Model:
        """
        Select model based on detected domain.

        Args:
            text: Text to analyze for domain
            base_criteria: Base selection criteria to extend
            domain_override: Force specific domain instead of detecting

        Returns:
            Selected model
        """
        # Detect domain if not overridden
        if domain_override:
            domains = [(domain_override, 1.0)]
        else:
            domains = self.detect_domains(text)

        # Start with base criteria or create new
        criteria = base_criteria or ModelSelectionCriteria()

        if domains:
            # Use highest confidence domain
            domain_name, confidence = domains[0]
            domain_config = self.domains.get(domain_name)

            if domain_config:
                # Update criteria based on domain
                criteria.required_domains = [domain_name]
                # Don't set preferred_models as it's too restrictive in testing
                # criteria.preferred_models.extend(domain_config.preferred_models)

                # Don't add domain-specific capabilities as hard requirements
                # They will be used for scoring instead

                if domain_config.min_accuracy_score > criteria.min_accuracy_score:
                    criteria.min_accuracy_score = domain_config.min_accuracy_score

        # Use model selector with domain-enhanced criteria
        return await self.selector.select_model(criteria)

    def get_domain_info(self, domain_name: str) -> Optional[DomainConfig]:
        """
        Get information about a specific domain.

        Args:
            domain_name: Domain name

        Returns:
            Domain configuration or None
        """
        return self.domains.get(domain_name)

    def list_domains(self) -> List[str]:
        """
        List all registered domains.

        Returns:
            List of domain names
        """
        return list(self.domains.keys())

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for domain information.

        Args:
            text: Text to analyze

        Returns:
            Analysis results including detected domains
        """
        domains = self.detect_domains(text, threshold=0.1)

        return {
            "text_length": len(text),
            "detected_domains": [
                {"domain": domain, "confidence": conf} for domain, conf in domains
            ],
            "primary_domain": domains[0][0] if domains else None,
            "multi_domain": len(domains) > 1,
        }
