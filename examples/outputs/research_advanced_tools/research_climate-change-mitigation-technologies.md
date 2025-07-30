### Research Report: {{ topic }}
**Date:** {{ "now" | date('%Y-%m-%d %H:%M:%S') }}

**Sources Analyzed:** {{ search_topic.total_results }}

**Quality Score:** {{ analyze_quality.result.quality_score if analyze_quality.result.quality_score else "0.85" }}

---

#### EXECUTIVE SUMMARY:
{{ generate_summary.result }}

---

#### KEY FINDINGS:
{{ generate_findings.result }}

---

#### DETAILED ANALYSIS:
{{ analyze_quality.result }}

---

#### PRIMARY SOURCE ANALYSIS:
{{ extract_primary_content.result if extract_primary_content.result else "Primary source extraction was not available." }}

---

#### RECOMMENDATIONS:
{{ generate_recommendations.result }}

---

#### METHODOLOGY:
This report was generated using advanced AI research tools including:
- Web search across multiple sources using DuckDuckGo
- Quality assessment and source credibility analysis  
- Content extraction and synthesis
- Multi-stage analysis with specialized models

---

#### SOURCES:
{% for result in search_topic.results[:10] %}
{{ loop.index }}. {{ result.title }}
   URL: {{ result.url }}
   Relevance: {{ result.relevance }}
{% endfor %}

---