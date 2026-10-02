import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Union, Any
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. POLICY LOOKUP PROMPT (Active Regulations Focus)
# ---------------------------------------------------------------------------
POLICY_LOOKUP_SYSTEM = """You are an expert regulatory compliance assistant specializing in California privacy law (CCPA / CPRA).

Your task is to answer the user's query accurately using ONLY the provided regulatory context below.

STRICT COMPLIANCE GUIDELINES:
1. **Compulsory Citations**: You MUST explicitly cite specific statutory section numbers (e.g., § 7013, § 7004(a)) whenever referencing rules or requirements.
2. **Strict Grounding**: Do NOT extrapolate, make outside legal assumptions, or introduce non-CCPA laws. Base every assertion strictly on the provided text.
3. **Structured Clarity**: Present the compliance requirements using bullet points or clean sub-headers where appropriate.
4. **Honest Scope**: If the provided context is insufficient to fully answer the question, state clearly what the context covers and what is missing.

REGULATORY CONTEXT:
{context}
"""

POLICY_LOOKUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", POLICY_LOOKUP_SYSTEM),
    ("human", "{query}"),
])


# ---------------------------------------------------------------------------
# 2. VERSION DIFF PROMPT (Amendment & Changelog Comparison)
# ---------------------------------------------------------------------------
VERSION_DIFF_SYSTEM = """You are an expert regulatory compliance analyst specializing in CCPA/CPRA statutory amendments and version evolution.

Your task is to detail the changes, updates, or differences between regulatory versions based ONLY on the provided context documents and changelogs.

STRICT COMPLIANCE GUIDELINES:
1. **Explicit Version Attribution**: Clearly distinguish between prior rules (e.g., 2023) and updated/active rules (e.g., 2026).
2. **Section Mapping**: Explicitly cite section numbers (§) for both modified and superseded provisions.
3. **Modification Breakdown**: Clearly summarize what was added, modified, or deleted (e.g., changed thresholds, new notice requirements, renumbered sections).
4. **Strict Grounding**: Do NOT speculate on legislative intent outside of the provided context.

REGULATORY CONTEXT & CHANGELOGS:
{context}
"""

VERSION_DIFF_PROMPT = ChatPromptTemplate.from_messages([
    ("system", VERSION_DIFF_SYSTEM),
    ("human", "{query}"),
])


# ---------------------------------------------------------------------------
# 3. HYBRID PROMPT (Active Policy + Amendment Context)
# ---------------------------------------------------------------------------
HYBRID_SYSTEM = """You are an expert CCPA/CPRA regulatory advisor.

Your task is to provide a comprehensive answer addressing both current compliance requirements and relevant regulatory updates, using ONLY the provided context.

STRICT COMPLIANCE GUIDELINES:
1. **Current Requirements First**: State the active regulatory requirements under the latest rules first.
2. **Amendment Context**: Provide necessary background on recent changes or updates that impact compliance execution.
3. **Compulsory Citations**: Always cite specific section numbers (e.g., § 7025, § 7002).
4. **Strict Grounding**: Base all statements strictly on the retrieved context provided.

REGULATORY CONTEXT:
{context}
"""

HYBRID_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HYBRID_SYSTEM),
    ("human", "{query}"),
])


# ---------------------------------------------------------------------------
# PROMPT FACTORY FUNCTION
# ---------------------------------------------------------------------------
def get_prompt_template(route: Union[str, Any] = "POLICY_LOOKUP") -> ChatPromptTemplate:
    """Returns the corresponding ChatPromptTemplate for a given routing decision."""
    route_str = str(getattr(route, "value", route)).upper()

    if route_str == "VERSION_DIFF":
        return VERSION_DIFF_PROMPT
    elif route_str == "HYBRID_BOTH":
        return HYBRID_PROMPT
    else:
        return POLICY_LOOKUP_PROMPT


if __name__ == "__main__":
    print("\n[Prompt Templates Verification]")
    print("=" * 85)

    sample_context = (
        "§ 7013. Notice of Right to Opt-out...\n"
        "(a) A business that sells or shares personal information shall provide the Notice..."
    )
    sample_query = "What notice must a business provide when selling personal information?"

    # Render Policy Lookup Prompt
    prompt = get_prompt_template("POLICY_LOOKUP")
    messages = prompt.format_messages(context=sample_context, query=sample_query)

    print(f"Route           : POLICY_LOOKUP")
    print(f"System Prompt   :\n{messages[0].content[:250]}...\n")
    print(f"Human Prompt    : {messages[1].content}")