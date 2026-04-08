
# ================================================================
# TYPE-SAFE OUTPUT SCRUBBER
# Replaces output_scrubber_node() in enterprise_merged_complete.py
# ================================================================
#
# THE BUG IN THE PREVIOUS VERSION
# ─────────────────────────────────────────────────────────────
# The old scrubber did:
#
#   if isinstance(last, AIMessage):
#       cleaned = layer3_output_scrubber(last.content)   ← WRONG
#
# last.content is typed as:  str | list[str | dict]
# Passing a list to re.sub() raises TypeError at runtime.
# Your IDE is correct to flag it.
#
# FULL TYPE PICTURE (from langchain_core.messages)
# ─────────────────────────────────────────────────────────────
# AIMessage.content: str | list[str | dict]
#
# When content is a LIST, each element is one of:
#
# 1. str              — plain text fragment (rare, older format)
#
# 2. dict with type="text"
#      {"type": "text", "text": "actual string here"}
#      ← standard OpenAI/Anthropic text block
#
# 3. dict with type="tool_use" / "tool_result"
#      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
#      ← tool invocation block (Anthropic) — NO text to scrub
#
# 4. dict with type="thinking" / "reasoning"  (Anthropic extended thinking)
#      {"type": "thinking", "thinking": "...", "signature": "..."}
#      ← MAY contain PII in thinking text — should be scrubbed
#
# 5. dict with type="image_url"
#      {"type": "image_url", "image_url": {"url": "..."}}
#      ← no text scrubbing needed
#
# 6. dict with type="image" / "audio" / "file"
#      multimodal blocks — skip, no text to scrub
#
# SCRUBBING RULES:
#   str content           → scrub directly
#   list[str | dict]:
#     element is str      → scrub it
#     element["type"] == "text"      → scrub element["text"]
#     element["type"] == "thinking"  → scrub element["thinking"]
#     anything else                  → pass through unchanged
#
# ================================================================

import re
import copy
from typing import Union

from langchain_core.messages import BaseMessage, AIMessage



# ── PII redaction patterns ──────────────────────────────────────
from defines import OUTPUT_REDACTION_RULES as OUTPUT_REDACT

def _scrub_str(text: str) -> str:
    """Apply all PII redaction patterns to a plain string."""
    for pattern, replacement in OUTPUT_REDACT:
        text = re.sub(pattern, replacement, text)
    return text


def _scrub_content_block(block: Union[str, dict]) -> Union[str, dict]:
    """
    Scrub a single content block element.

    Handles ALL known block types correctly:
    - str            → scrub and return str
    - dict type=text → scrub dict["text"], return updated dict copy
    - dict type=thinking → scrub dict["thinking"], return updated dict copy
    - dict type=tool_use / tool_result / image_url / image / audio / file
                     → return unchanged (no user-visible text to scrub)
    - dict unknown   → return unchanged (forward-compatible)

    Always returns the same type as the input.
    """
    if isinstance(block, str):
        return _scrub_str(block)

    if isinstance(block, dict):
        block_type = block.get("type", "")

        if block_type == "text":
            # Standard text block — scrub the "text" field
            raw = block.get("text", "")
            if not isinstance(raw, str):
                return block   # malformed block — leave untouched
            scrubbed = _scrub_str(raw)
            if scrubbed == raw:
                return block   # no change — return original (avoids copy overhead)
            return {**block, "text": scrubbed}   # shallow copy with patched field

        if block_type == "thinking":
            # Anthropic extended thinking block — scrub the "thinking" field
            raw = block.get("thinking", "")
            if not isinstance(raw, str):
                return block
            scrubbed = _scrub_str(raw)
            if scrubbed == raw:
                return block
            return {**block, "thinking": scrubbed}

        # All other block types: tool_use, tool_result, image_url,
        # image, audio, file, computer_call_output, etc.
        # These are either structured data or binary content — leave untouched.
        return block

    # Not str or dict — unknown type, return unchanged
    return block


def scrub_message_content(
    content: Union[str, list],
) -> Union[str, list]:
    """
    Scrub ALL PII from an AIMessage content field.

    Correctly handles:
        str                  → scrub and return str
        list[str | dict]     → scrub each element individually,
                               preserve block types faithfully
    """
    if isinstance(content, str):
        return _scrub_str(content)

    if isinstance(content, list):
        scrubbed_blocks = []
        changed = False
        for block in content:
            scrubbed = _scrub_content_block(block)
            scrubbed_blocks.append(scrubbed)
            if scrubbed is not block:   # identity check — cheap change detection
                changed = True
        # Only allocate a new list if something actually changed
        return scrubbed_blocks if changed else content

    # Unexpected type (future-proof) — return unchanged
    return content


def scrub_ai_message(message: AIMessage, trace_id: str = "") -> AIMessage:
    """
    Return a new AIMessage with PII scrubbed from ALL content fields.

    Uses model_copy (Pydantic v2) to avoid mutating the original message.
    Appends the trace_id suffix ONLY to the text that the user will read
    (the last text-bearing element), not to structured tool blocks.

    Returns the SAME message object if no changes were made (zero-copy fast path).
    """
    original_content = message.content
    scrubbed_content = scrub_message_content(original_content)

    # Append trace reference suffix
    suffix = f"_Ref: {trace_id[:8]}_" if trace_id else ""

    if suffix:
        if isinstance(scrubbed_content, str):
            scrubbed_content = scrubbed_content + suffix
        elif isinstance(scrubbed_content, list):
            # Find the LAST text-type block and append there.
            # Do NOT append to tool_use or image blocks.
            new_list = list(scrubbed_content)   # shallow copy
            appended = False
            for i in range(len(new_list) - 1, -1, -1):
                block = new_list[i]
                if isinstance(block, str):
                    new_list[i] = block + suffix
                    appended = True
                    break
                if isinstance(block, dict) and block.get("type") == "text":
                    new_list[i] = {**block, "text": block["text"] + suffix}
                    appended = True
                    break
            if not appended:
                # No text block found — append as a new text block
                new_list.append({"type": "text", "text": suffix.strip()})
            scrubbed_content = new_list

    # Fast path: if nothing changed, return the original message unchanged
    if scrubbed_content is original_content and not suffix:
        return message

    return message.model_copy(update={"content": scrubbed_content})