class InferenceError(Exception):
    """Anything this service could not complete.

    Carries no request payload: the inputs here are candidate CVs and job
    descriptions, and an exception message is the easiest way for personal data
    to end up somewhere it should not be.
    """


class DocumentExtractionError(InferenceError):
    pass


class RetrievalError(InferenceError):
    pass


class KnowledgeSourceError(InferenceError):
    pass
