from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search enterprise knowledge base for policies and FAQs."""
    return f"[KB] Policy for '{query}': Requires 2-factor approval for amounts > $10,000."

@tool
def get_financial_summary(account_id: str) -> str:
    """Retrieve financial summary for an account."""
    return f"[Finance] Account {account_id}: Balance $42,500. Last tx: 2026-04-01."

@tool
def create_support_ticket(issue: str, priority: str = "medium") -> str:
    """Create a support ticket in the enterprise ticketing system."""
    ticket_id = f"TKT-{abs(hash(issue)) % 10000:04d}"
    return f"Ticket {ticket_id} created, priority={priority}."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human agent."""
    return f"Escalated. A human will respond within 15 minutes. Reason: {reason}"
