from unittest.mock import MagicMock


def make_response(text: str):
    """Wrap a plain string into the shape OpenAIPromptedSimulator expects."""
    resp = MagicMock()
    # simulate .choices[0].message.content == text
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp
