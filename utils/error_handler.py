"""Errors the user can understand. Anything else is shown as a generic message."""


class DocuMindError(Exception):
    """Base class. The message is safe to show to end users."""


class UnsupportedFileError(DocuMindError):
    pass


class FileTooLargeError(DocuMindError):
    pass


class EmptyDocumentError(DocuMindError):
    pass


class CorruptedFileError(DocuMindError):
    pass


class UploadLimitError(DocuMindError):
    pass


class DuplicateSourceError(DocuMindError):
    pass


class InvalidYouTubeURLError(DocuMindError):
    pass


class TranscriptUnavailableError(DocuMindError):
    pass


class EmbeddingError(DocuMindError):
    pass


class VectorStoreError(DocuMindError):
    pass


class LLMError(DocuMindError):
    pass


class MissingAPIKeyError(LLMError):
    pass


def friendly_message(error):
    """Message for end users. Never exposes a traceback."""
    if isinstance(error, DocuMindError):
        return str(error)
    return "Something unexpected went wrong. Please try again."
