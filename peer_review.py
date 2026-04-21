"""
Peer Review Simulation - Simulated peer review for research papers.

Phase 5.2: Peer Review Simulation.
"""

import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class ReviewAspect(Enum):
    """Aspects reviewed in peer review."""

    ORIGINALITY = "originality"
    TECHNICAL_QUALITY = "technical_quality"
    CLARITY = "clarity"
    REPRODUCIBILITY = "reproducibility"
    SIGNIFICANCE = "significance"
    RELATED_WORK = "related_work"
    EXPERIMENTAL_DESIGN = "experimental_design"
    STATISTICAL_RIGOR = "statistical_rigor"


class ReviewVerdict(Enum):
    """Peer review verdict."""

    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"


class ReviewStrength(Enum):
    """Strength level."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class ReviewCriterion:
    """A single review criterion."""

    aspect: ReviewAspect
    strength: ReviewStrength
    comment: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect.value,
            "strength": self.strength.value,
            "comment": self.comment,
        }


@dataclass
class ReviewerProfile:
    """Simulated reviewer profile."""

    name: str
    expertise: List[str]
    strictness: float = 1.0  # 0.5 to 1.5


@dataclass
class PeerReview:
    """A simulated peer review."""

    id: str
    paper_title: str
    reviewer: ReviewerProfile
    verdict: ReviewVerdict
    summary: str
    criteria: List[ReviewCriterion] = field(default_factory=list)
    confidential_comments: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    score: int = 0  # 1-5 overall score


@dataclass
class ReviewSimulationConfig:
    """Configuration for review simulation."""

    num_reviewers: int = 3
    seed: Optional[int] = None
    reviewer_pool: List[ReviewerProfile] = field(default_factory=list)
    acceptance_threshold: float = 0.6


# Simulated reviewer pool
DEFAULT_REVIEWERS = [
    ReviewerProfile(
        name="Dr. Anonymous Reviewer 1",
        expertise=["machine_learning", "optimization"],
        strictness=1.0,
    ),
    ReviewerProfile(
        name="Dr. Anonymous Reviewer 2",
        expertise=["NLP", "transformers"],
        strictness=1.1,
    ),
    ReviewerProfile(
        name="Dr. Anonymous Reviewer 3",
        expertise=["autoML", "neural_architecture_search"],
        strictness=0.9,
    ),
    ReviewerProfile(
        name="Dr. Anonymous Reviewer 4",
        expertise=["reinforcement_learning", "agent_systems"],
        strictness=1.2,
    ),
    ReviewerProfile(
        name="Dr. Anonymous Reviewer 5",
        expertise=["scientific_computing", "optimization"],
        strictness=0.95,
    ),
]

# Review templates by aspect
REVIEW_TEMPLATES = {
    ReviewAspect.ORIGINALITY: {
        ReviewStrength.STRONG: [
            "This work presents a novel approach to {topic} that differs significantly from existing methods.",
            "The contribution is highly original and opens new research directions.",
            "The ideas presented are fresh and not previously explored in the literature.",
        ],
        ReviewStrength.MODERATE: [
            "The work builds on existing ideas but applies them in a new context.",
            "Some aspects are novel, though the core concept draws from prior work.",
            "The approach combines existing methods in a reasonable way.",
        ],
        ReviewStrength.WEAK: [
            "The work is largely incremental without significant novelty.",
            "Similar ideas have appeared in recent literature.",
            "The contribution is not sufficiently distinct from prior work.",
        ],
        ReviewStrength.NONE: [
            "This work does not present novel contributions.",
            "The ideas are entirely derivative of existing papers.",
        ],
    },
    ReviewAspect.TECHNICAL_QUALITY: {
        ReviewStrength.STRONG: [
            "The technical approach is sound and well-executed.",
            "The methodology is rigorous and the math is correct.",
            "Excellent technical depth and precision throughout.",
        ],
        ReviewStrength.MODERATE: [
            "The technical details are mostly correct with minor issues.",
            "The approach is reasonable, though some details need clarification.",
            "The methodology is adequate for the claims.",
        ],
        ReviewStrength.WEAK: [
            "Technical details are lacking or incorrect in places.",
            "There are concerns about the technical soundness.",
            "More rigorous technical treatment is needed.",
        ],
        ReviewStrength.NONE: [
            "The technical approach has fundamental flaws.",
            "The methodology is unsound.",
        ],
    },
    ReviewAspect.REPRODUCIBILITY: {
        ReviewStrength.STRONG: [
            "The paper provides complete details to reproduce all results.",
            "Code and data are available, making full replication possible.",
            "Excellent reproducibility - all experiments can be verified.",
        ],
        ReviewStrength.MODERATE: [
            "Most details are provided, with minor gaps.",
            "The paper provides sufficient information for reproduction.",
            "Key details are present, though some are missing.",
        ],
        ReviewStrength.WEAK: [
            "Critical details are missing for reproduction.",
            "Not enough information to verify the claims.",
            "The experimental setup is unclear.",
        ],
        ReviewStrength.NONE: [
            "The paper cannot be reproduced from the given information.",
            "Insufficient details for any reproduction.",
        ],
    },
}


class PeerReviewSimulator:
    """Simulates peer review process."""

    def __init__(self, config: Optional[ReviewSimulationConfig] = None):
        self.config = config or ReviewSimulationConfig()
        self.reviews: List[PeerReview] = []

        if self.config.seed:
            random.seed(self.config.seed)

    def _get_topics(self, paper_content: str) -> List[str]:
        """Extract topics from paper content."""
        topics = []
        keywords = {
            "machine learning": "ML",
            "language model": "language modeling",
            "transformer": "transformers",
            "optimization": "optimization",
            "training": "training",
            "autonomous": "autonomous systems",
            "neural": "neural networks",
        }

        for keyword, topic in keywords.items():
            if keyword in paper_content.lower():
                topics.append(topic)

        return topics or ["machine learning"]

    def _determine_strength(
        self,
        aspect: ReviewAspect,
        quality_score: float,
        strictness: float,
    ) -> ReviewStrength:
        """Determine strength level based on quality and strictness."""
        adjusted = quality_score * strictness

        if adjusted >= 0.8:
            return ReviewStrength.STRONG
        elif adjusted >= 0.6:
            return ReviewStrength.MODERATE
        elif adjusted >= 0.4:
            return ReviewStrength.WEAK
        else:
            return ReviewStrength.NONE

    def _generate_criterion(
        self,
        aspect: ReviewAspect,
        reviewer: ReviewerProfile,
        topics: List[str],
        quality: float,
    ) -> ReviewCriterion:
        """Generate a single review criterion."""
        # Determine strength based on quality and reviewer strictness
        strength = self._determine_strength(aspect, quality, reviewer.strictness)

        # Get template comment
        templates = REVIEW_TEMPLATES.get(aspect, {}).get(strength, ["No comment."])
        comment = random.choice(templates).format(topic=random.choice(topics))

        return ReviewCriterion(aspect=aspect, strength=strength, comment=comment)

    def simulate_review(
        self,
        paper_title: str,
        paper_content: str,
        metrics: Dict[str, Any],
    ) -> PeerReview:
        """Simulate a single peer review."""
        # Select reviewer
        reviewer = random.choice(self.config.reviewer_pool or DEFAULT_REVIEWERS)

        # Extract topics
        topics = self._get_topics(paper_content)

        # Calculate quality scores based on metrics
        base_scores = {
            ReviewAspect.ORIGINALITY: metrics.get("originality", 0.7),
            ReviewAspect.TECHNICAL_QUALITY: metrics.get("technical_quality", 0.75),
            ReviewAspect.CLARITY: metrics.get("clarity", 0.8),
            ReviewAspect.REPRODUCIBILITY: metrics.get("reproducibility", 0.85),
            ReviewAspect.SIGNIFICANCE: metrics.get("significance", 0.7),
            ReviewAspect.RELATED_WORK: metrics.get("related_work", 0.75),
            ReviewAspect.EXPERIMENTAL_DESIGN: metrics.get("experimental_design", 0.8),
            ReviewAspect.STATISTICAL_RIGOR: metrics.get("statistical_rigor", 0.75),
        }

        # Generate criteria
        criteria = []
        for aspect, score in base_scores.items():
            criterion = self._generate_criterion(aspect, reviewer, topics, score)
            criteria.append(criterion)

        # Determine verdict based on scores
        avg_score = sum(base_scores.values()) / len(base_scores)

        if avg_score >= 0.75:
            verdict = ReviewVerdict.ACCEPT
        elif avg_score >= 0.6:
            verdict = ReviewVerdict.MINOR_REVISION
        elif avg_score >= 0.45:
            verdict = ReviewVerdict.MAJOR_REVISION
        else:
            verdict = ReviewVerdict.REJECT

        # Calculate overall score (1-5)
        score = int(avg_score * 5)

        # Generate summary
        summaries = {
            ReviewVerdict.ACCEPT: [
                "This paper presents a strong contribution to the field. Accept.",
                "Excellent work with clear merit. Accept.",
            ],
            ReviewVerdict.MINOR_REVISION: [
                "A solid paper with Minor Revision requested.",
                "Good work, minor issues to address.",
            ],
            ReviewVerdict.MAJOR_REVISION: [
                "Major Revision required to address significant concerns.",
                "The paper has merit but needs substantial work.",
            ],
            ReviewVerdict.REJECT: [
                "Unfortunately, this paper cannot be accepted in current form.",
                "The paper does not meet the bar for publication.",
            ],
        }
        summary = random.choice(summaries[verdict])

        review = PeerReview(
            id=f"review_{len(self.reviews) + 1}",
            paper_title=paper_title,
            reviewer=reviewer,
            verdict=verdict,
            summary=summary,
            criteria=criteria,
            score=score,
        )

        self.reviews.append(review)
        return review

    def simulate_review_round(
        self,
        paper_title: str,
        paper_content: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> List[PeerReview]:
        """
        Simulate a full review round with multiple reviewers.

        Args:
            paper_title: Title of the paper
            paper_content: Content of the paper
            metrics: Quality metrics (optional, will be generated if not provided)

        Returns:
            List of peer reviews
        """
        # Generate metrics if not provided
        if metrics is None:
            # Default reasonable metrics
            metrics = {
                "originality": random.uniform(0.6, 0.9),
                "technical_quality": random.uniform(0.65, 0.9),
                "clarity": random.uniform(0.7, 0.9),
                "reproducibility": random.uniform(0.7, 0.9),
                "significance": random.uniform(0.6, 0.85),
                "related_work": random.uniform(0.65, 0.9),
                "experimental_design": random.uniform(0.7, 0.9),
                "statistical_rigor": random.uniform(0.65, 0.9),
            }

        # Generate reviews
        num = self.config.num_reviewers
        reviews = []

        for _ in range(num):
            review = self.simulate_review(paper_title, paper_content, metrics)
            reviews.append(review)

        return reviews

    def get_consensus(self, reviews: List[PeerReview]) -> Dict[str, Any]:
        """Calculate consensus from multiple reviews."""
        if not reviews:
            return {}

        # Count verdicts
        verdicts = {}
        for r in reviews:
            v = r.verdict.value
            verdicts[v] = verdicts.get(v, 0) + 1

        # Average score
        avg_score = sum(r.score for r in reviews) / len(reviews)

        # Agreement level
        unique_verdicts = len(verdicts)
        if unique_verdicts == 1:
            agreement = "unanimous"
        elif unique_verdicts == len(reviews):
            agreement = "divergent"
        else:
            agreement = "majority"

        return {
            "verdicts": verdicts,
            "average_score": avg_score,
            "agreement": agreement,
            "num_reviews": len(reviews),
        }

    def generate_review_report(
        self,
        paper_title: str,
        reviews: List[PeerReview],
    ) -> str:
        """Generate a formatted review report."""
        lines = [
            "# Peer Review Report",
            "",
            f"**Paper:** {paper_title}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "---",
            "",
        ]

        for i, review in enumerate(reviews, 1):
            lines.append(f"## Review {i}: {review.reviewer.name}")
            lines.append(f"**Verdict:** {review.verdict.value.upper()}")
            lines.append(f"**Score:** {review.score}/5")
            lines.append("")
            lines.append(f"**Summary:** {review.summary}")
            lines.append("")

            lines.append("### Criteria:")
            for criterion in review.criteria:
                lines.append(
                    f"- **{criterion.aspect.value}:** {criterion.strength.value}"
                )
                lines.append(f"  - {criterion.comment}")
            lines.append("")

        # Add consensus
        consensus = self.get_consensus(reviews)
        lines.append("---")
        lines.append("")
        lines.append("## Consensus")
        lines.append(f"Average Score: {consensus['average_score']:.1f}/5")
        lines.append(f"Agreement: {consensus['agreement']}")

        return "\n".join(lines)


def run_peer_review_example():
    """Run an example peer review simulation."""
    print("=== Peer Review Simulation Example ===\n")

    # Sample paper content
    paper_title = "Autonomous Research for LLM Training"
    paper_content = """
    We present an autonomous research system that improves language model training.
    Our approach uses multi-agent systems with machine learning optimization.
    The system achieves significant improvements through automated experimentation.
    """

    # Quality metrics for the paper
    metrics = {
        "originality": 0.82,
        "technical_quality": 0.78,
        "clarity": 0.85,
        "reproducibility": 0.90,
        "significance": 0.75,
        "related_work": 0.80,
        "experimental_design": 0.82,
        "statistical_rigor": 0.78,
    }

    # Create simulator
    config = ReviewSimulationConfig(
        num_reviewers=3,
        seed=42,
    )
    simulator = PeerReviewSimulator(config)

    # Run review round
    reviews = simulator.simulate_review_round(paper_title, paper_content, metrics)

    # Print results
    print(f"Paper: {paper_title}\n")

    for i, review in enumerate(reviews, 1):
        print(f"Review {i}: {review.reviewer.name}")
        print(f"  Verdict: {review.verdict.value}")
        print(f"  Score: {review.score}/5")
        print(f"  Summary: {review.summary}")
        print()

    # Consensus
    consensus = simulator.get_consensus(reviews)
    print(f"Consensus: {consensus['agreement']}")
    print(f"Average Score: {consensus['average_score']:.1f}/5")

    # Generate report
    report = simulator.generate_review_report(paper_title, reviews)
    print("\n--- Full Report Preview ---")
    print(report[:800] + "...")


if __name__ == "__main__":
    run_peer_review_example()
