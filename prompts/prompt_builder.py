from config import settings
from prompts.prompt_version_store import record_prompt_usage

MODE_CLOSED_BOOK = "closed_book"
MODE_OPEN_BOOK = "open_book"

def get_instruction(variant=None):
    variant = settings.INSTRUCTION_VARIANT if variant is None else variant
    if variant not in settings.INSTRUCTION_VARIANTS:
        raise settings.ConfigurationError(f"Unknown instruction variant: {variant!r}")
    return settings.INSTRUCTION_VARIANTS[variant], variant

def _context_to_text(context):
    if not context:
        return ""
    if isinstance(context, (list, tuple)):
        return "\n".join(str(item).strip() for item in context if str(item).strip())
    return str(context).strip()

def get_slot_key(test_case):
    """Prompt-version identity. Prefers the dataset's own ID, which survives
    sheet consolidation and row reordering; falls back to the original
    category+position scheme for rows loaded without an ID."""

    row_id = test_case.get("id")

    if row_id:
        return str(row_id)

    return f"{test_case.get('category')}:{test_case.get('category_position')}"

def build_prompt(test_case, instruction_variant=None, send_context=None):
    """Returns:
        {
            "text":                the assembled prompt string,
            "mode":                "closed_book" | "open_book",
            "context_in_prompt":   bool,
            "instruction_variant": str,
            "slot_key":            str,
            "version":             int | None,
        }

    `context_in_prompt` is the field that matters downstream: it is what makes
    the answer-vs-context metrics applicable or not.
    """

    question = str(test_case.get("input") or "").strip()

    if send_context is None:
        send_context = settings.SEND_CONTEXT_IN_PROMPT

    context_text = _context_to_text(test_case.get("context"))
    context_in_prompt = bool(context_text) and bool(send_context)

    instruction, variant = get_instruction(instruction_variant)

    sections = []

    if context_in_prompt:
        if instruction:
            sections.append(instruction)
        sections.append(f"Reference material:\n{context_text}")
        sections.append(f"Question: {question}")
        mode = MODE_OPEN_BOOK
    else:
        sections.append(question)
        mode = MODE_CLOSED_BOOK

    text = "\n\n".join(section for section in sections if section)

    slot_key = get_slot_key(test_case)

    entry = record_prompt_usage(
        slot_key,
        text,
        category=test_case.get("category"),
    )

    return {
        "text": text,
        "mode": mode,
        "context_in_prompt": context_in_prompt,
        "instruction_variant": variant if context_in_prompt else None,
        "instruction": instruction if context_in_prompt else "",
        "slot_key": slot_key,
        "version": (entry or {}).get("version"),
    }
