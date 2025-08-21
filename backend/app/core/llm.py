import os, json
from .settings import settings

# We keep the LLM optional. If provider is 'none' or key missing, we no-op.

def enrich_findings_with_llm(findings: list[dict], context: dict) -> list[dict]:
    if settings.LLM_PROVIDER == "none":
        return findings

    model = os.getenv("LLM_MODEL_ID", settings.LLM_MODEL_ID)
    provider = settings.LLM_PROVIDER
    api_key = settings.OPENROUTER_API_KEY if provider == "openrouter" else settings.OPENAI_API_KEY
    if not api_key:
        return findings

    try:
        from litellm import completion
        system = "You are a senior code reviewer. For each finding, provide a concise rationale and a concrete remediation suggestion. Respond in JSON list matching input order with keys 'rationale' and 'remediation'."
        user = json.dumps({"project_context": context, "findings": findings[:20]}, indent=2)  # cap batch
        resp = completion(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.2)
        content = resp.choices[0].message["content"]
        parsed = json.loads(content)
        for i, extra in enumerate(parsed):
            if i < len(findings):
                findings[i]["remediation"] = extra.get("remediation") or findings[i].get("remediation")
        return findings
    except Exception:
        return findings
