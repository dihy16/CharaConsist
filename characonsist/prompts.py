"""Prompt construction and cumulative token-span bookkeeping."""

import re


ROLE_TAGS = {"S": "subject", "A": "predicate", "O": "object", "R": "recipient"}
_ROLE_TAG_PATTERN = re.compile(r"\[(/?)([SAOR])\]")
_BINDING_TAG_PATTERN = re.compile(r"\[(/?)([CA])(\d+)\]")


def get_text_tokens_length(pipe, text):
    """Return the non-padding token count, excluding the terminal token."""
    text_mask = pipe.tokenizer_2(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).attention_mask
    return max(0, text_mask.sum().item() - 1)


def build_prompt_and_spans(background, foreground, action, pipe):
    """Build the FLUX prompt and return cumulative text-token boundaries."""
    prompt, background_end, action_start, real_end, _ = build_prompt_spans_and_roles(
        background, foreground, action, pipe
    )
    return prompt, background_end, action_start, real_end


def parse_indexed_tags(text, expected_kind):
    """Strip indexed C/A tags and return clean-text spans keyed by entity id."""
    clean_parts = []
    spans = {}
    active = None
    active_start = None
    source_pos = 0
    clean_pos = 0
    for match in _BINDING_TAG_PATTERN.finditer(text):
        chunk = text[source_pos:match.start()]
        clean_parts.append(chunk)
        clean_pos += len(chunk)
        closing, kind, entity_id = match.groups()
        if kind != expected_kind:
            raise ValueError(
                f"[{kind}{entity_id}] is not valid in the {expected_kind}-tagged section."
            )
        if closing:
            if active != entity_id:
                raise ValueError(f"Unexpected closing tag {match.group(0)}.")
            if clean_pos == active_start:
                raise ValueError(f"{kind}{entity_id} tag must not be empty.")
            spans[entity_id] = (active_start, clean_pos)
            active = None
            active_start = None
        else:
            if active is not None:
                raise ValueError("Indexed binding tags must not be nested.")
            if entity_id in spans:
                raise ValueError(f"{kind}{entity_id} may appear only once per section.")
            active = entity_id
            active_start = clean_pos
        source_pos = match.end()
    clean_parts.append(text[source_pos:])
    if active is not None:
        raise ValueError(f"Unclosed {expected_kind}{active} tag.")
    return "".join(clean_parts), spans


def parse_role_tags(action):
    """Strip optional role tags and return their character spans in clean text."""
    clean_parts = []
    role_char_spans = {}
    active_role = None
    active_start = None
    source_pos = 0
    clean_pos = 0

    for match in _ROLE_TAG_PATTERN.finditer(action):
        text = action[source_pos:match.start()]
        clean_parts.append(text)
        clean_pos += len(text)
        closing, short_name = match.groups()
        role = ROLE_TAGS[short_name]
        if closing:
            if active_role != role:
                raise ValueError(f"Unexpected closing role tag {match.group(0)}.")
            if clean_pos == active_start:
                raise ValueError(f"Role tag {short_name} must not be empty.")
            role_char_spans[role] = (active_start, clean_pos)
            active_role = None
            active_start = None
        else:
            if active_role is not None:
                raise ValueError("Role tags must not be nested.")
            if role in role_char_spans:
                raise ValueError(f"Role tag {short_name} may appear only once per action.")
            active_role = role
            active_start = clean_pos
        source_pos = match.end()

    tail = action[source_pos:]
    clean_parts.append(tail)
    if active_role is not None:
        raise ValueError(f"Unclosed role tag for {active_role}.")
    clean_action = "".join(clean_parts)
    return clean_action, role_char_spans


def build_prompt_spans_and_roles(background, foreground, action, pipe):
    """Build a clean prompt plus cumulative section and optional role spans."""
    prompt, background_end, action_start, real_end, role_token_spans, _ = (
        build_prompt_spans_roles_and_bindings(background, foreground, action, pipe)
    )
    return prompt, background_end, action_start, real_end, role_token_spans


def build_prompt_spans_roles_and_bindings(background, foreground, action, pipe):
    """Build a clean prompt with role spans and indexed character/action spans."""
    clean_foreground, character_char_spans = parse_indexed_tags(foreground, "C")
    indexed_action, action_char_spans = parse_indexed_tags(action, "A")
    clean_action, role_char_spans = parse_role_tags(indexed_action)
    if role_char_spans and action_char_spans:
        raise ValueError("Role tags and indexed action tags cannot be combined in one action.")
    background_prefix = background + " "
    foreground_prefix = clean_foreground + " "
    foreground_end = background_prefix + foreground_prefix
    prompt = foreground_end + clean_action

    background_end = get_text_tokens_length(pipe, background_prefix)
    action_start = get_text_tokens_length(pipe, foreground_end)
    real_end = get_text_tokens_length(pipe, prompt)

    # A prompt truncated before its action has an empty, neutral action span.
    action_start = min(action_start, real_end)
    role_token_spans = {}
    for role, (start, end) in role_char_spans.items():
        token_start = get_text_tokens_length(pipe, foreground_end + clean_action[:start])
        token_end = get_text_tokens_length(pipe, foreground_end + clean_action[:end])
        token_start = min(max(action_start, token_start), real_end)
        token_end = min(max(token_start, token_end), real_end)
        if token_start < token_end:
            role_token_spans[role] = (token_start, token_end)
    binding_spans = {"characters": {}, "actions": {}}
    for entity_id, (start, end) in character_char_spans.items():
        token_start = get_text_tokens_length(pipe, background_prefix + clean_foreground[:start])
        token_end = get_text_tokens_length(pipe, background_prefix + clean_foreground[:end])
        token_start = min(max(background_end, token_start), action_start)
        token_end = min(max(token_start, token_end), action_start)
        if token_start < token_end:
            binding_spans["characters"][entity_id] = (token_start, token_end)
    for entity_id, (start, end) in action_char_spans.items():
        token_start = get_text_tokens_length(pipe, foreground_end + clean_action[:start])
        token_end = get_text_tokens_length(pipe, foreground_end + clean_action[:end])
        token_start = min(max(action_start, token_start), real_end)
        token_end = min(max(token_start, token_end), real_end)
        if token_start < token_end:
            binding_spans["actions"][entity_id] = (token_start, token_end)
    return prompt, background_end, action_start, real_end, role_token_spans, binding_spans
