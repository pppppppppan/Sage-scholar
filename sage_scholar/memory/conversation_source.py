"""Bounded original conversation snapshots, independent of Context compression."""


def conversation_messages(request, answer=None):
    # Keep a contiguous suffix of complete messages. Never truncate a sentence and
    # accidentally remove a rejection/qualification from a proposal or confirmation.
    recent, size = [], 0
    history = request.history
    for index in range(len(history) - 1, max(-1, len(history) - 7), -1):
        message = history[index]
        if message.get('role') not in {'user', 'assistant'}:
            continue
        content = message.get('content', '')
        if size + len(content) > 24000:
            break
        recent.append({'role': message['role'], 'content': content,
            'message_id': message.get('message_id') or message.get('id'),
            'history_index': index, 'extraction_eligible': False})
        size += len(content)
    messages = list(reversed(recent))
    messages.append({'role': 'user', 'content': request.query,
        'message_id': f'{request.request_id}:user' if request.request_id else None,
        'history_index': len(history), 'extraction_eligible': True})
    if answer is not None:
        messages.append({'role': 'assistant', 'content': answer,
            'message_id': f'{request.request_id}:assistant' if request.request_id else None,
            'history_index': len(history) + 1, 'extraction_eligible': False})
    return messages
