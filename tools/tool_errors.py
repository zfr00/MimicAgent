"""Tool execution exceptions."""


class ExemplarPrefixError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Exemplar prefix unavailable: {result.get('error', 'invalid response')}")


def require_exemplar_prefix(result, success=True, *, count=None):
    if not success or not isinstance(result, dict) or result.get('error') or not result.get('exemplars'):
        raise ExemplarPrefixError(result if isinstance(result, dict) else {'error': 'Invalid recall response'})
    if count is not None and (not isinstance(result['exemplars'], list) or len(result['exemplars']) != count):
        raise ExemplarPrefixError({'error': f'Expected {count} prefix exemplars', 'code': 'invalid_prefix'})
    return result
