INSIGHT_PROMPT = """
You are an expert product analyst. Analyze the following customer reviews and extract structured insights.

From the reviews provided, identify:
1. top_issues: List of top 3-5 issues customers are facing (be specific)
2. patterns: List of 2-3 patterns or trends you notice across reviews
3. confidence_score: Your confidence in these insights (0.0 to 1.0)

Return ONLY a valid JSON object. No explanation, no markdown.

Example response:
{
  "top_issues": ["delivery delays during peak hours", "payment refund not processed"],
  "patterns": ["most complaints occur on weekends", "tier-2 cities face more issues"],
  "confidence_score": 0.87
}
"""