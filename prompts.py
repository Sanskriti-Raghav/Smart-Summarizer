def build_prompt(text: str, summary_words: int, takeaways_count: int) -> str:
    """
    Constructs a strict, JSON-only prompt. Allows takeaways_count to be 0
    (used internally during map step).
    """
    template = f"""You are a precise writing assistant.
Task:
1) Provide a concise, faithful summary of the input text.
2) List exactly {takeaways_count} key takeaways as short bullet points.
Length guidance: aim for about {summary_words} words.
Output JSON ONLY with keys: summary, key_takeaways (array).
Text:
{text}
"""
    return template
