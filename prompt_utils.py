"""Prompt construction and cumulative token-span bookkeeping."""


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
    background_prefix = background + " "
    foreground_prefix = foreground + " "
    foreground_end = background_prefix + foreground_prefix
    prompt = foreground_end + action

    background_end = get_text_tokens_length(pipe, background_prefix)
    action_start = get_text_tokens_length(pipe, foreground_end)
    real_end = get_text_tokens_length(pipe, prompt)

    # A prompt truncated before its action has an empty, neutral action span.
    action_start = min(action_start, real_end)
    return prompt, background_end, action_start, real_end
