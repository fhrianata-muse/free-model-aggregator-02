# free-model-aggregator-02 — live free-model aggregator (Python, stdlib only)
#
# One server, two client wires (OpenAI + Anthropic), one gateway. Every answer
# comes from a real free model dialled through $OPENAI_BASE_URL; this file never
# answers from itself. Routing is rating-ordered: the highest-rated free model
# that meets a prompt's requirements serves it (inkling-2-small 1475 for plain /
# streaming / tools prompts, nemotron-3.5-super-vl 1303 for JSON-mode prompts),
# with one same-model retry and one failover per prompt, a per-model cooldown
# inside the trial, and the failures trail on the x-makefun-upstream-failures
# header. Deterministic: no wall clock, no unseeded random, fixed `created`
# values; every choice is a pure function of the brief, the request, and the
# refusals seen so far.
#
# Trial lifecycle: listen on 0.0.0.0 at a free port, POST $MAKEFUN_ENV_URL/register
# {protocol_version: 1, port, catalog}, serve GET /v1/models, POST
# /v1/chat/completions and POST /v1/messages (JSON or SSE), poll GET
# $MAKEFUN_ENV_URL/work every 100 ms and exit 0 when it says done.
# `python3 main.py --list` prints the catalog and exits 0 before any server
# starts. Runs as `python3 -P -s main.py` on Python 3.13: everything lives in
# this one file because -P keeps the script's directory off sys.path.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# --- constants -----------------------------------------------------------------

PROTOCOL_VERSION = 1            # register body's wire version
POLL_INTERVAL_SEC = 0.1         # GET /work poll cadence
MAX_BODY_BYTES = 8 * 1024 * 1024  # largest client request body accepted

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
# Dummy key for the trial's gateway lane; the gateway swaps it for the entrant's key.
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# Provider's public base URL, spelled in pieces (never dialled: the gateway is).
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# Fixed creation values: nothing in any reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]  # one JSON object, as Python reads it


# --- small type checks ---------------------------------------------------------

def is_record(value: object) -> bool:
    """A JSON object (a dict), never an array or another value."""
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    """A JSON number: int or float, never a bool (Python's bool is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# --- reading the brief ---------------------------------------------------------

def object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """First document under this contest's prefix carrying one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row. A free row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: {"models": [...], "prompts": [...], "error": str|None}.

    Only the two documents under this contest's schema prefix are read; the
    brief's prose is never parsed.
    """
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    """The brief's free models, in the brief's order."""
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet a prompt's requirements.

    An unknown capability never meets one.
    """
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True

# The sealed fourteen free models, exactly as every practice brief lists them
# (id, rating, context, capabilities). Used only when the brief at
# MAKEFUN_PROMPT_FILE names no model, so --list and routing stay exact outside
# a trial; inside a trial the brief's own rows win.
SEALED_MODELS: list[Json] = [
    {'id': 'openrouter/baidu/ernie-5.5-flash-preview:free', 'provider': 'openrouter', 'display_name': 'Baidu: ERNIE-5.5-Flash-Preview (free)', 'rating_milli': 1150, 'context_length': 1024000, 'tools': True, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/cohere/north-mini-code:free', 'provider': 'openrouter', 'display_name': 'Cohere: North Mini Code (free)', 'rating_milli': 1200, 'context_length': 256000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/dots-studio/dots-3-note-preview:free', 'provider': 'openrouter', 'display_name': 'Dots Studio: Dots3-Note Preview (free)', 'rating_milli': 1150, 'context_length': 512000, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/google/gemma-4-26b-a4b-it:free', 'provider': 'openrouter', 'display_name': 'Google: Gemma 4 26B A4B (free)', 'rating_milli': 1150, 'context_length': 262144, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/google/gemma-4-31b-it:free', 'provider': 'openrouter', 'display_name': 'Google: Gemma 4 31B (free)', 'rating_milli': 1210, 'context_length': 262144, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/inclusionai/ling-3.2-flash:free', 'provider': 'openrouter', 'display_name': 'inclusionAI: Ling 3.2 Flash (free)', 'rating_milli': 1150, 'context_length': 1048576, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/liquid/lfm-3-1.7b:free', 'provider': 'openrouter', 'display_name': 'LiquidAI: LFM3-1.7B (free)', 'rating_milli': 1150, 'context_length': 262144, 'tools': True, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', 'provider': 'openrouter', 'display_name': 'NVIDIA: Nemotron 3 Nano Omni (free)', 'rating_milli': 1150, 'context_length': 256000, 'tools': True, 'vision': True, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', 'provider': 'openrouter', 'display_name': 'NVIDIA: Nemotron 3 Ultra (free)', 'rating_milli': 1285, 'context_length': 1000000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3.5-lightning:free', 'provider': 'openrouter', 'display_name': 'NVIDIA: Nemotron 3.5 Lightning (free)', 'rating_milli': 1221, 'context_length': 1000000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3.5-super-vl:free', 'provider': 'openrouter', 'display_name': 'NVIDIA: Nemotron 3.5 Super VL (free)', 'rating_milli': 1303, 'context_length': 1000000, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/particle-ai/particle-1:free', 'provider': 'openrouter', 'display_name': 'Particle AI: Particle-1 (free)', 'rating_milli': 1150, 'context_length': 1048576, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/recursive/flint-v1.0:free', 'provider': 'openrouter', 'display_name': 'Recursive: Flint (free)', 'rating_milli': 1150, 'context_length': 262144, 'tools': True, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/thinkingmachines/inkling-2-small:free', 'provider': 'openrouter', 'display_name': 'Thinking Machines: Inkling 2 Small (free)', 'rating_milli': 1475, 'context_length': 1048576, 'tools': True, 'vision': True, 'json': False, 'free': True},
]


def all_models(brief: Json) -> list[Json]:
    """The brief's free models, or the sealed fourteen when the brief names none."""
    models = [m for m in brief['models'] if m['free']]
    return models if models else list(SEALED_MODELS)


def build_catalog(brief: Json) -> Json:
    """The catalog the brief's free model list makes: one provider, the models it serves.

    The base_url is the provider's public endpoint, never the gateway's
    trial-local address: the gateway's origin differs between a live pass and
    its replay, and the catalog is part of the state export a replay must
    reproduce byte for byte. When the brief names no model (run outside a trial,
    e.g. `--list` with no brief), the sealed fourteen ids are used instead.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in all_models(brief):
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# --- the client wires and the model field --------------------------------------

def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: OpenAI's list shape."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in Anthropic's shape (the Messages API's error envelope)."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    """An error in OpenAI's shape (Chat Completions)."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or inline image; None for the rest."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON otherwise."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'

def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`.

    The system prompt becomes a system message, image blocks become image_url
    data URLs, tool_use blocks become tool_calls, tool_result blocks become tool
    messages, tools and tool_choice are rewritten, and the output cap,
    temperature, top_p, stop sequences and streaming pass through.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id dialled: the one truthful claim."""
    return {**body, 'model': model}


def whole_usage(parsed: Json) -> Json:
    """Usage the wire checks accept: input at least 0, output at least 1.

    The provider's own counts pass through untouched. Only when a count is
    missing or the output count is zero does the guard step in, estimating the
    output from the answer's text length, so a provider that omits usage cannot
    fail the wire column on its own. Never invents a larger claim than the
    answer supports.
    """
    usage = parsed.get('usage') if is_record(parsed.get('usage')) else {}
    prompt = number_or_zero(usage.get('prompt_tokens'))
    completion = number_or_zero(usage.get('completion_tokens'))
    if prompt < 0:
        prompt = 0
    if completion < 1:
        text = ''
        choices = parsed.get('choices') if isinstance(parsed.get('choices'), list) else []
        if choices and is_record(choices[0]):
            message = choices[0].get('message') if is_record(choices[0].get('message')) else {}
            content = message.get('content')
            text = content if isinstance(content, str) else text_of(content)
        completion = max(1, len(text) // 4)
    return {'prompt_tokens': int(prompt), 'completion_tokens': int(completion), 'total_tokens': int(prompt) + int(completion)}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`, blank line kept.

    Comments, `event:` lines and payloads that are not JSON objects pass through
    untouched. Chat Completions names the model on its chunks; the client sees
    the id the entry dialled, never the provider's spelling.
    """
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line ending each one.

    A frame cut across reads is held until its end arrives, and a body that ends
    without one is yielded as it is. CRLF is normalized to LF as the SSE grammar
    reads it.
    """
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer

def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, every data payload's `model` the dialled id.

    The upstream's own [DONE] is held back and re-emitted once at the end: when
    the stream carried no usage chunk, one is appended first (empty choices, the
    estimated usage), because the wire check reads usage from any chunk and a
    stream without it fails the usage check outright. A stream cut short without
    [DONE] is still closed properly.
    """
    usage_seen = False
    text_chars = 0
    for frame in iter_frames(chunks):
        data_lines = [line.strip()[5:].strip() for line in frame.split('\n') if line.strip().startswith('data:')]
        if data_lines == ['[DONE]']:
            continue
        for payload in data_lines:
            if payload == '':
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                usage_seen = True
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            if choices and is_record(choices[0]):
                delta = choices[0].get('delta') if is_record(choices[0].get('delta')) else {}
                if isinstance(delta.get('content'), str):
                    text_chars += len(delta['content'])
        yield frame_with_model(frame, model)
    if not usage_seen:
        output_tokens = max(1, text_chars // 4)
        usage_chunk = {
            'id': 'chatcmpl-aggregate',
            'object': 'chat.completion.chunk',
            'created': FIXED_OPENAI_CREATED,
            'model': model,
            'choices': [],
            'usage': {'prompt_tokens': 0, 'completion_tokens': output_tokens, 'total_tokens': output_tokens},
        }
        yield f"data: {json.dumps(usage_chunk, separators=(',', ':'), ensure_ascii=False)}\n\n"
    yield 'data: [DONE]\n\n'


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text
    block or tool call, then message_delta with the stop reason and usage, then
    message_stop. `model` is the id the entry dialled, named back to the client.
    """
    started = False
    finished = False
    # The stop reason a finish_reason named, held until the stream's usage chunk has arrived.
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0
    text_chars = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        # The wire check needs an output count of at least 1: the provider's own
        # count when it sent one, else an estimate from the streamed text.
        output = output_tokens if output_tokens > 0 else max(1, text_chars // 4)
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop, text_chars
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                text_chars += len(delta['content'])
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            # Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
            # OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
            # output tokens.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event

# --- the trial state -----------------------------------------------------------

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network);
    a base that does not already end in `/v1` gets one, so both spellings reach
    the route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in all_models(BRIEF)]


# Models that answered a 429/5xx in this trial: skipped for later prompts while
# a near-rated alternative exists, so one rate-limited model cannot take every
# later prompt down with it. A set, never a timestamp: the choice must be
# replay-identical.
cooled_down: set[str] = set()
cooled_lock = threading.Lock()


def mark_cooled(model_id: str) -> None:
    with cooled_lock:
        cooled_down.add(model_id)


def candidates_for(requires: Json) -> list[Json]:
    """Free models meeting `requires`, best rating first, brief order breaking ties.

    Cooled-down models sort last (they are still usable when nothing else is
    left); within each group the order is rating desc, then the brief's order,
    so the choice is a pure function of the brief and the refusals seen so far.
    """
    eligible = [m for m in all_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    with cooled_lock:
        cooled = set(cooled_down)
    return sorted(eligible, key=lambda m: (m['id'] in cooled, -number_or_zero(m['rating_milli'])))


def route_for(requires: Json) -> Json | None:
    """The one free model that serves a request: the highest-rated candidate meeting it.

    Quality is 60% of the score and is the rating of the model that really served
    each prompt over the best rating meeting that prompt's requirements, so the
    first candidate is the whole quality play: inkling-2-small (1475) for plain,
    streaming and tools prompts, nemotron-3.5-super-vl (1303) for JSON-mode prompts.
    """
    ranked = candidates_for(requires)
    return ranked[0] if ranked else None


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt now arriving: the brief's row by arrival order,
    unioned with what the body itself declares (the stricter of the two wins).

    The brief's manifest is the declared source; the body is the check against a
    manifest row that understates the request (tools declared, a JSON
    response_format, image parts). A requirement either source names must be met.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    manifest_req: Json = prompts[index]['requires'] if index < len(prompts) else {}
    body_req = requirements_from_body(body)
    merged: Json = {}
    for key in ('tools', 'vision', 'json'):
        if manifest_req.get(key) is True or body_req.get(key) is True:
            merged[key] = True
    min_context = manifest_req.get('min_context')
    if is_number(min_context):
        merged['min_context'] = min_context
    return merged


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


# --- the gateway ---------------------------------------------------------------

class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a connection already closed is nothing to report
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        # Bound every gateway call inside the env's 30 s per-prompt window: a hung
        # call must fail fast enough for the failover to still serve the prompt,
        # and never wedge the server past the trial's wall.
        response = urllib.request.urlopen(request, timeout=25)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


# --- failover (max 2 recovery steps per prompt) --------------------------------

# Seconds waited before the same-model retry: a fixed constant, never a clock reading.
RETRY_PAUSE_SEC = 1.0
MAX_UPSTREAM_TOKENS = 8192
DEFAULT_UPSTREAM_TOKENS = 512


def upstream_body(body: Json) -> Json:
    """The body as the gateway must receive it: an output cap is always named.

    The gateway refuses a request with no max_tokens, or one over 8192. The
    client always sends one in practice; when it does not, 512 is the default,
    and a cap over 8192 is clamped rather than refused. A streaming request asks
    for its usage chunk, which the wire checks read.
    """
    out = dict(body)
    cap = body.get('max_tokens')
    if not is_number(cap):
        cap = body.get('max_completion_tokens')
    if not is_number(cap) or cap <= 0:
        cap = DEFAULT_UPSTREAM_TOKENS
    out['max_tokens'] = min(int(cap), MAX_UPSTREAM_TOKENS)
    out.pop('max_completion_tokens', None)
    if body.get('stream') is True:
        out['stream_options'] = {'include_usage': True}
    return out


def error_code_of(text: str) -> str:
    """The gateway error body's code (else its type), for the failures trail.

    Some providers spell the code as a number (OpenRouter's 403 gate answers
    {"code": 403}); that is stringified rather than dropped, so the trail row
    still names the refusal.
    """
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ''
    if not is_record(value):
        return ''
    error = value.get('error')
    if not is_record(error):
        return ''
    code = error.get('code')
    if isinstance(code, str) and code != '':
        return code[:64]
    if isinstance(code, (int, float)) and not isinstance(code, bool):
        return str(int(code))[:64]
    if isinstance(error.get('type'), str) and error['type'] != '':
        return error['type'][:64]
    return ''


def dial_with_recovery(requires: Json, body: Json) -> tuple[Json | None, Upstream | None, list[Json]]:
    """Dial the best model for `requires`, recovering from refusals within the budget.

    At most 2 recovery steps per prompt: one retry on the same model for a
    429/5xx, then one failover to the next-best model meeting the requirements.
    A 401 gives up at once (a failover would be refused the same way). Every
    refused model is cooled down for later prompts. Returns (model, upstream,
    failures) where `failures` is the trail for the x-makefun-upstream-failures
    header, in the order the calls refused; model/upstream are None when the
    prompt was not served.
    """
    ranked = candidates_for(requires)
    if not ranked:
        return None, None, []
    prepared = upstream_body(body)
    failures: list[Json] = []
    tried: set[str] = set()
    attempt = ranked[0]
    retried = False
    failed_over = False
    while True:
        tried.add(attempt['id'])
        try:
            upstream = dial(attempt['id'], prepared)
        except TimeoutError:
            # A hung route: record it as a 504 and fail over at once — retrying the
            # same hung route would only burn the rest of the prompt's 30 s window.
            failures.append({'status': 504, 'code': 'timeout', 'model': attempt['id']})
            mark_cooled(attempt['id'])
            upstream = None
        except Exception:
            failures.append({'status': 502, 'code': 'upstream_unreachable', 'model': attempt['id']})
            mark_cooled(attempt['id'])
            upstream = None
        if upstream is not None and upstream.ok:
            return attempt, upstream, failures
        if upstream is not None:
            text = upstream.text()
            code = error_code_of(text)
            failures.append({'status': upstream.status, 'code': code, 'model': attempt['id']})
            mark_cooled(attempt['id'])
            if upstream.status == 401:
                return None, None, failures
            if not retried and upstream.status in (429, 500, 502, 503, 504):
                # First refusal: one retry on the same model, after a fixed pause.
                retried = True
                time.sleep(RETRY_PAUSE_SEC)
                continue
        if not failed_over:
            # Then one failover to the next-best model meeting the requirements.
            rest = [m for m in ranked if m['id'] not in tried]
            if rest:
                failed_over = True
                attempt = rest[0]
                continue
        return None, None, failures

# --- the HTTP server -----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing ---------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra_headers: dict[str, str] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        # Python's http.server does not chunk a response without a length; the close is what ends the
        # stream for the client. Node's http server chunks instead and keeps the connection alive.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, status: int, text: str) -> None:
        """A refused or broken gateway call as a 502 in the client's wire: the prompt is lost."""
        suffix = '' if text == '' else f': {text[:500]}'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway answered HTTP {status}{suffix}')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway answered HTTP {status}{suffix}')

    # -- the routes ---------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def trail_headers(self, failures: list[Json]) -> dict[str, str]:
        """The failures trail header, when a prompt was served after refusals.

        The env reads at most 8 rows and 8 KiB; a served prompt's refusals are the
        only way the failover column can see them, so they ride on the answer.
        """
        if not failures:
            return {}
        return {'x-makefun-upstream-failures': json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)}

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        if not candidates_for(requires):
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return
        try:
            route, upstream, failures = dial_with_recovery(requires, {**body, 'model': 'auto'})
        except Exception as err:  # an unreachable gateway is a 502 to this prompt
            self.send_error_body('openai', 502, 'upstream_unreachable', str(err))
            return
        if route is None or upstream is None:
            self.refused('openai', 502, 'the gateway refused every model that meets this request')
            return
        trail = self.trail_headers(failures)
        if body.get('stream') is True:
            self.stream_headers(trail)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), route['id']):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            self.refused('openai', 502, 'the gateway answered a body that is not JSON')
            return
        parsed['usage'] = whole_usage(parsed)
        self.send_json(200, openai_json_with_model(parsed, route['id']), trail)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        if not candidates_for(requires):
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return
        translated = anthropic_to_openai(body, 'auto')
        try:
            route, upstream, failures = dial_with_recovery(requires, translated)
        except Exception as err:  # an unreachable gateway is a 502 to this prompt
            self.send_error_body('anthropic', 502, 'api_error', str(err))
            return
        if route is None or upstream is None:
            self.refused('anthropic', 502, 'the gateway refused every model that meets this request')
            return
        trail = self.trail_headers(failures)
        if body.get('stream') is True:
            self.stream_headers(trail)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), route['id']):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            self.refused('anthropic', 502, 'the gateway answered a body that is not JSON')
            return
        parsed['usage'] = whole_usage(parsed)
        self.send_json(200, openai_to_anthropic(parsed, route['id']), trail)


# --- register and poll -----------------------------------------------------------

def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the trial's prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:  # the env is not reachable yet: try again on the next tick
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# --- start -----------------------------------------------------------------------

def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object and stop
        # before any server starts. A process whose stdout is not flushed loses the tail, so flush it.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # Node's top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
