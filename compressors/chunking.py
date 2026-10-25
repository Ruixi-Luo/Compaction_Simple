"""只拆分超长模型输入，不丢字符、不看问题；正常长度输入原样保留。"""


def split_overlong(text, count_tokens, limit):
    """优先在中间附近的空白处二分；极长无空白串按字符二分。"""
    if count_tokens(text) <= limit:
        return [text]
    if len(text) < 2:
        raise ValueError('A single character exceeds the configured token limit.')
    middle = len(text) // 2
    boundaries = [i + 1 for i, char in enumerate(text[:-1]) if char.isspace()]
    cut = min(boundaries, key=lambda i: abs(i - middle)) if boundaries else middle
    return (split_overlong(text[:cut], count_tokens, limit)
            + split_overlong(text[cut:], count_tokens, limit))
