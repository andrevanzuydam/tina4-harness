"""Ollama wire adapter for the shared model server; no task-specific prompting."""
import base64
import json
import os
import tempfile


class Channels:
    """Incrementally separate model channels, including delimiters split across tokens."""
    markers = {'<think>': True, '</think>': False,
               '<|channel>thought\n': True, '<channel|>': False}

    def __init__(self, emit, thinking=False):
        self.emit, self.thinking, self.pending = emit, thinking, ''

    def feed(self, chunk, final=False):
        self.pending += chunk
        while self.pending:
            found = [(self.pending.find(m), m) for m in self.markers if m in self.pending]
            if found:
                at, marker = min(found)
                if at:
                    self.emit(self.pending[:at], self.thinking)
                self.thinking = self.markers[marker]
                self.pending = self.pending[at + len(marker):]
                continue
            keep = 0
            if not final:
                for marker in self.markers:
                    for length in range(1, min(len(marker), len(self.pending)) + 1):
                        if self.pending.endswith(marker[:length]):
                            keep = max(keep, length)
            ready = self.pending[:-keep] if keep else self.pending
            if ready:
                self.emit(ready, self.thinking)
            self.pending = self.pending[-keep:] if keep else ''
            break


def chat(handler, body, model, lock, generate=False):
    messages = body.get('messages') or [{'role': 'user', 'content': body.get('prompt', '')}]
    options = body.get('options') or {}
    limit = int(options.get('num_predict') or 4096)
    if not 1 <= limit <= 32768:
        raise ValueError('num_predict must be between 1 and 32768')
    think = bool(body.get('think', True))
    stream = bool(body.get('stream', True))
    answer, reasoning = [], []
    def frame(content='', thinking='', done=False, **extra):
        value = {'model': model.repo_id, 'message': {'role': 'assistant', 'content': content}, 'done': done, **extra}
        if thinking:
            value['message']['thinking'] = thinking
        if generate:
            value['response'] = content
            value['thinking'] = thinking
        return value
    def send(value):
        handler._write_chunk((json.dumps(value) + '\n').encode())
    def emit(text, thinking):
        (reasoning if thinking else answer).append(text)
        if stream:
            send(frame(thinking=text) if thinking else frame(content=text))
    if stream:
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/x-ndjson')
        handler.send_header('Transfer-Encoding', 'chunked')
        handler.end_headers()
    try:
        with lock:
            images = [image for message in messages for image in message.get('images', [])]
            prompt = model._render(messages, think)
            parser = Channels(emit, prompt.rfind('<think>') > prompt.rfind('</think>'))
            if images:
                parser = Channels(emit)  # describe() renders its own multimodal template
                raw = images[-1].split(',', 1)[-1]
                fd, path = tempfile.mkstemp(suffix='.png')
                try:
                    with os.fdopen(fd, 'wb') as file:
                        file.write(base64.b64decode(raw, validate=True))
                    task = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
                    parser.feed(model.describe(path, task, max_tokens=limit, temperature=options.get('temperature', 0)))
                finally:
                    os.unlink(path)
            else:
                model.generate(messages, on_text=parser.feed, max_tokens=limit,
                               temperature=options.get('temperature', 0), enable_thinking=think,
                               raw_prompt=body.get('prompt') if generate and body.get('raw') else None)
            parser.feed('', final=True)
            count = model.last_stats.generation_tokens
            result = frame(done=True, done_reason='length' if count >= limit else 'stop', eval_count=count)
        if stream:
            send(result)
        else:
            result['message']['content'] = ''.join(answer)
            result['message']['thinking'] = ''.join(reasoning)
            handler._send_json(result)
    except Exception as error:
        if stream:
            send({'error': str(error), 'done': True})
        else:
            raise
    finally:
        if stream:
            handler._write_chunk(b'')
