# free-model-aggregator-02 — an aggregator over OpenRouter's free models.
#
# One process serves two client wires (OpenAI chat completions and Anthropic
# messages) by dialling the make.fun gateway, which is the only route to a
# model. Nothing here invents an answer: every reply body is a provider
# answer passed through, stamped with the id actually dialled. For each
# prompt the router picks the best-rated free model whose capabilities cover
# the prompt (JSON mode, tools, images, context), retries it once and fails
# over once when the gateway refuses, and reports recovered refusals on the
# x-makefun-upstream-failures header. All choices are pure functions of the
# brief, the request, and the refusals seen in this trial, so a replay picks
# the same models. The server registers with the trial env, serves until the
# env reports the work done, then exits 0.

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

# --- settings ----------------------------------------------------------------

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
DIAL_TIMEOUT_SEC = 25
RETRY_PAUSE_SEC = 1.0
MAX_UPSTREAM_TOKENS = 8192
DEFAULT_UPSTREAM_TOKENS = 512

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
# The trial lane's stand-in key; the gateway exchanges it for the entrant's key.
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The provider's public endpoint, named for the catalog (never dialled: all
# model calls go through the gateway above).
OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# Replies never read the clock: creation markers are fixed.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]


# --- reading values out of JSON ----------------------------------------------

def is_obj(value: object) -> bool:
    """True for a JSON object; arrays and scalars are not objects."""
    return isinstance(value, dict)


def is_num(value: object) -> bool:
    """True for a JSON number; a bool is not a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_str(value: object, default: str = '') -> str:
    """A string value, or `default` when the value is not a string."""
    return value if isinstance(value, str) else default


def as_num(value: object) -> int | float | None:
    """A finite JSON number, or None."""
    if not is_num(value):
        return None
    return value if math.isfinite(value) else None


def as_num_or_zero(value: object) -> int | float:
    """A JSON number, or 0."""
    return value if is_num(value) else 0


def as_bool(value: object) -> bool | None:
    """A JSON boolean, or None when the value is not one."""
    return value if isinstance(value, bool) else None


# --- the brief ---------------------------------------------------------------

def documents_in(text: str) -> list[Json]:
    """Each JSON object in `text` that carries a string `schema`, in order.

    The brief is prose with JSON documents embedded in it, so every `{` is
    tried as the start of one document; a position that does not decode is
    skipped and the scan moves one character on.
    """
    decoder = json.JSONDecoder()
    found: list[Json] = []
    pos = 0
    while True:
        start = text.find('{', pos)
        if start == -1:
            return found
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if is_obj(value) and isinstance(value.get('schema'), str):
            found.append(value)
        pos = start + 1


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix holding one of `keys` as a list."""
    for doc in docs:
        if SCHEMA_PREFIX not in as_str(doc.get('schema')):
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row. A free row is routed as the gateway spells its id."""
    if not is_obj(value):
        return None
    raw = as_str(value.get('id')) or as_str(value.get('model'))
    if raw == '':
        return None
    provider = as_str(value.get('provider')) or as_str(value.get('wire'))
    model_id = raw
    if provider == 'openrouter' and not raw.lower().startswith('openrouter/'):
        model_id = 'openrouter/' + raw
    rating = as_num(value.get('rating_milli'))
    row: Json = {}
    row['id'] = model_id
    row['provider'] = provider if provider != '' else 'openrouter'
    row['display_name'] = as_str(value.get('display_name')) or raw
    row['rating_milli'] = rating if rating is not None else 0
    row['context_length'] = as_num(value.get('context_length'))
    row['tools'] = as_bool(value.get('tools'))
    row['vision'] = as_bool(value.get('vision'))
    row['json'] = as_bool(value.get('json'))
    row['free'] = value.get('free') is not False
    return row


def prompt_of(value: object) -> Json | None:
    """One prompt row."""
    if not is_obj(value):
        return None
    prompt_id = as_str(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_obj(value.get('requires')) else {}
    needed: Json = {}
    for key in ('tools', 'vision', 'json'):
        if requires.get(key) is True:
            needed[key] = True
    min_context = requires.get('min_context')
    if is_num(min_context):
        needed['min_context'] = min_context
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': needed,
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


def meets_requirements(requires: Json, model: Json) -> bool:
    """Whether a model's capabilities meet a prompt's requirements.

    An unknown capability never meets one.
    """
    for key in ('tools', 'vision', 'json'):
        if requires.get(key) is True and model.get(key) is not True:
            return False
    min_context = requires.get('min_context')
    if min_context is not None:
        length = model.get('context_length')
        if length is None or length < min_context:
            return False
    return True


# --- the sealed catalog ------------------------------------------------------

# The fourteen free models, exactly as every practice brief lists them. Used
# only when the brief at MAKEFUN_PROMPT_FILE names no model, so --list and
# routing stay exact outside a trial; inside a trial the brief's own rows win.
SEALED_MODELS: list[Json] = [
    {
        'id': 'openrouter/baidu/ernie-5.5-flash-preview:free',
        'provider': 'openrouter',
        'display_name': 'Baidu: ERNIE-5.5-Flash-Preview (free)',
        'rating_milli': 1150,
        'context_length': 1024000,
        'tools': True,
        'vision': False,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/cohere/north-mini-code:free',
        'provider': 'openrouter',
        'display_name': 'Cohere: North Mini Code (free)',
        'rating_milli': 1200,
        'context_length': 256000,
        'tools': True,
        'vision': False,
        'json': False,
        'free': True,
    },
    {
        'id': 'openrouter/dots-studio/dots-3-note-preview:free',
        'provider': 'openrouter',
        'display_name': 'Dots Studio: Dots3-Note Preview (free)',
        'rating_milli': 1150,
        'context_length': 512000,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/google/gemma-4-26b-a4b-it:free',
        'provider': 'openrouter',
        'display_name': 'Google: Gemma 4 26B A4B (free)',
        'rating_milli': 1150,
        'context_length': 262144,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/google/gemma-4-31b-it:free',
        'provider': 'openrouter',
        'display_name': 'Google: Gemma 4 31B (free)',
        'rating_milli': 1210,
        'context_length': 262144,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/inclusionai/ling-3.2-flash:free',
        'provider': 'openrouter',
        'display_name': 'inclusionAI: Ling 3.2 Flash (free)',
        'rating_milli': 1150,
        'context_length': 1048576,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/liquid/lfm-3-1.7b:free',
        'provider': 'openrouter',
        'display_name': 'LiquidAI: LFM3-1.7B (free)',
        'rating_milli': 1150,
        'context_length': 262144,
        'tools': True,
        'vision': False,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
        'provider': 'openrouter',
        'display_name': 'NVIDIA: Nemotron 3 Nano Omni (free)',
        'rating_milli': 1150,
        'context_length': 256000,
        'tools': True,
        'vision': True,
        'json': False,
        'free': True,
    },
    {
        'id': 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free',
        'provider': 'openrouter',
        'display_name': 'NVIDIA: Nemotron 3 Ultra (free)',
        'rating_milli': 1285,
        'context_length': 1000000,
        'tools': True,
        'vision': False,
        'json': False,
        'free': True,
    },
    {
        'id': 'openrouter/nvidia/nemotron-3.5-lightning:free',
        'provider': 'openrouter',
        'display_name': 'NVIDIA: Nemotron 3.5 Lightning (free)',
        'rating_milli': 1221,
        'context_length': 1000000,
        'tools': True,
        'vision': False,
        'json': False,
        'free': True,
    },
    {
        'id': 'openrouter/nvidia/nemotron-3.5-super-vl:free',
        'provider': 'openrouter',
        'display_name': 'NVIDIA: Nemotron 3.5 Super VL (free)',
        'rating_milli': 1303,
        'context_length': 1000000,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/particle-ai/particle-1:free',
        'provider': 'openrouter',
        'display_name': 'Particle AI: Particle-1 (free)',
        'rating_milli': 1150,
        'context_length': 1048576,
        'tools': True,
        'vision': True,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/recursive/flint-v1.0:free',
        'provider': 'openrouter',
        'display_name': 'Recursive: Flint (free)',
        'rating_milli': 1150,
        'context_length': 262144,
        'tools': True,
        'vision': False,
        'json': True,
        'free': True,
    },
    {
        'id': 'openrouter/thinkingmachines/inkling-2-small:free',
        'provider': 'openrouter',
        'display_name': 'Thinking Machines: Inkling 2 Small (free)',
        'rating_milli': 1475,
        'context_length': 1048576,
        'tools': True,
        'vision': True,
        'json': False,
        'free': True,
    },
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
    rows: list[Json] = []
    seen: set[str] = set()
    for model in all_models(brief):
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        entry: Json = {}
        entry['id'] = model_id
        entry['free'] = True
        entry['context_length'] = model['context_length']
        rows.append(entry)
    if not rows:
        return {'providers': []}
    provider: Json = {}
    provider['id'] = 'openrouter'
    provider['base_url'] = OPENROUTER_PUBLIC_BASE
    provider['key_env'] = OPENROUTER_KEY_ENV
    provider['wire'] = 'openai'
    provider['models'] = rows
    return {'providers': [provider]}


# --- the client wires and the model field ------------------------------------

def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: OpenAI's list shape."""
    data: list[Json] = []
    for m in models:
        row: Json = {}
        row['id'] = m['id']
        row['object'] = 'model'
        row['created'] = FIXED_OPENAI_CREATED
        row['owned_by'] = m['provider']
        data.append(row)
    return {'object': 'list', 'data': data}


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    data: list[Json] = []
    for m in models:
        row: Json = {}
        row['type'] = 'model'
        row['id'] = m['id']
        row['display_name'] = m['display_name']
        row['created_at'] = FIXED_ANTHROPIC_CREATED_AT
        data.append(row)
    return {'data': data, 'has_more': False}


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
    parts: list[str] = []
    for block in content:
        if is_obj(block) and block.get('type') == 'text' and isinstance(block.get('text'), str):
            parts.append(block['text'])
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or inline image; None for the rest."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    if block.get('type') != 'image':
        return None
    source = block.get('source')
    if not is_obj(source):
        return None
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
        if text != '':
            return text
        return json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_obj(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_obj(parsed):
            return parsed
    return {}


# OpenAI finish_reason -> Anthropic stop_reason; anything else ends the turn.
STOP_REASON_BY_FINISH = {
    'length': 'max_tokens',
    'tool_calls': 'tool_use',
    'function_call': 'tool_use',
    'content_filter': 'refusal',
}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    return STOP_REASON_BY_FINISH.get(finish, 'end_turn') if isinstance(finish, str) else 'end_turn'


def _tool_call_of(block: Json) -> Json:
    """One Anthropic tool_use block as an OpenAI tool call."""
    return {
        'id': block['id'] if isinstance(block.get('id'), str) else '',
        'type': 'function',
        'function': {
            'name': block['name'] if isinstance(block.get('name'), str) else '',
            'arguments': json.dumps(
                block['input'] if is_obj(block.get('input')) else {},
                separators=(',', ':'),
                ensure_ascii=False,
            ),
        },
    }


def _assistant_message(raw: Json) -> Json | None:
    """One Anthropic assistant turn as an OpenAI assistant message; None when it has no content."""
    content = raw.get('content')
    if isinstance(content, str):
        return {'role': 'assistant', 'content': content}
    if not isinstance(content, list):
        return None
    tool_calls: list[Json] = []
    for block in content:
        if is_obj(block) and block.get('type') == 'tool_use':
            tool_calls.append(_tool_call_of(block))
    message: Json = {'role': 'assistant', 'content': text_of(content) or None}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return message


def _user_messages(raw: Json) -> list[Json]:
    """One Anthropic user turn as OpenAI messages: tool results first, then the parts."""
    content = raw.get('content')
    if isinstance(content, str):
        return [{'role': 'user', 'content': content}]
    if not isinstance(content, list):
        return []
    out: list[Json] = []
    parts: list[Json] = []
    for block in content:
        if not is_obj(block):
            continue
        if block.get('type') == 'tool_result':
            out.append(
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
        out.append({'role': 'user', 'content': parts})
    return out


def _openai_tools(body: Json) -> list[Json]:
    """An Anthropic tools array as OpenAI function tools."""
    tools: list[Json] = []
    declared = body.get('tools')
    if not isinstance(declared, list):
        return tools
    for raw in declared:
        if not is_obj(raw) or not isinstance(raw.get('name'), str):
            continue
        schema = raw['input_schema'] if is_obj(raw.get('input_schema')) else {'type': 'object', 'properties': {}}
        tools.append(
            {
                'type': 'function',
                'function': {
                    'name': raw['name'],
                    'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                    'parameters': schema,
                },
            }
        )
    return tools


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
        if not is_obj(raw):
            continue
        if raw.get('role') == 'assistant':
            message = _assistant_message(raw)
            if message is not None:
                messages.append(message)
        else:
            messages.extend(_user_messages(raw))
    out: Json = {'model': model, 'messages': messages}
    for key in ('max_tokens', 'temperature', 'top_p'):
        if is_num(body.get(key)):
            out[key] = body[key]
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    tools = _openai_tools(body)
    if tools:
        out['tools'] = tools
    choice = body.get('tool_choice')
    if is_obj(choice):
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


def _tool_use_block(call: Json) -> Json | None:
    """One OpenAI tool call as an Anthropic tool_use block; None when it is not one."""
    if not is_obj(call.get('function')):
        return None
    fn = call['function']
    return {
        'type': 'tool_use',
        'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
        'name': fn['name'] if isinstance(fn.get('name'), str) else '',
        'input': arguments_of(fn.get('arguments')),
    }


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_obj(choices[0]) else {}
    message = choice.get('message') if is_obj(choice.get('message')) else {}
    content: list[Json] = []
    text = message.get('content')
    if not isinstance(text, str):
        text = text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_obj(call):
            continue
        block = _tool_use_block(call)
        if block is not None:
            content.append(block)
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_obj(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': as_num_or_zero(usage.get('prompt_tokens')),
            'output_tokens': as_num_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id dialled: the one truthful claim."""
    return {**body, 'model': model}


def _answer_text(parsed: Json) -> str:
    """The first choice's message text, for estimating usage when the provider omits it."""
    choices = parsed.get('choices') if isinstance(parsed.get('choices'), list) else []
    if not choices or not is_obj(choices[0]):
        return ''
    message = choices[0].get('message') if is_obj(choices[0].get('message')) else {}
    content = message.get('content')
    return content if isinstance(content, str) else text_of(content)


def whole_usage(parsed: Json) -> Json:
    """Usage the wire checks accept: input at least 0, output at least 1.

    The provider's own counts pass through untouched. Only when a count is
    missing or the output count is zero does the guard step in, estimating the
    output from the answer's text length, so a provider that omits usage cannot
    fail the wire column on its own. Never invents a larger claim than the
    answer supports.
    """
    usage = parsed.get('usage') if is_obj(parsed.get('usage')) else {}
    prompt = as_num_or_zero(usage.get('prompt_tokens'))
    completion = as_num_or_zero(usage.get('completion_tokens'))
    if prompt < 0:
        prompt = 0
    if completion < 1:
        completion = max(1, len(_answer_text(parsed)) // 4)
    return {'prompt_tokens': int(prompt), 'completion_tokens': int(completion), 'total_tokens': int(prompt) + int(completion)}


# --- server-sent events ------------------------------------------------------

def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    head = '' if event is None else f'event: {event}\n'
    return f'{head}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def sse_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line ending each one.

    Nothing is parsed from a single read: bytes accumulate until a whole line
    has arrived, because a read may end inside a CRLF pair or inside a
    multi-byte character, and a frame cut at either place must not be parsed
    half-formed. A line ends with LF, CRLF or CR; a blank line ends the frame.
    The frames yielded are their lines joined with LF, as the SSE grammar
    reads them. A body that ends mid-frame yields what arrived, as it stands.
    """
    pending = b''
    lines: list[str] = []

    def take_line(final: bool) -> bytes | None:
        """The next complete line in `pending`, without its ending; None while it is still arriving."""
        nonlocal pending
        lf = pending.find(b'\n')
        cr = pending.find(b'\r')
        if lf == -1 and cr == -1:
            return None
        if cr != -1 and (lf == -1 or cr < lf):
            # A CR ends the line, alone or as the first half of a CRLF pair; a
            # CR that is the buffer's last byte may still gain its LF half
            # from the next read, so it completes no line until then.
            if cr + 1 == len(pending) and not final:
                return None
            width = 2 if cr + 1 < len(pending) and pending[cr + 1:cr + 2] == b'\n' else 1
            cut = cr
        else:
            cut, width = lf, 1
        line = pending[:cut]
        pending = pending[cut + width:]
        return line

    for chunk in chunks:
        pending += chunk.encode('utf-8') if isinstance(chunk, str) else chunk
        while True:
            line = take_line(False)
            if line is None:
                break
            if line == b'':
                yield '\n'.join(lines)
                lines = []
            else:
                lines.append(line.decode('utf-8', 'replace'))
    while True:
        line = take_line(True)
        if line is None:
            break
        if line == b'':
            yield '\n'.join(lines)
            lines = []
        else:
            lines.append(line.decode('utf-8', 'replace'))
    if pending != b'':
        lines.append(pending.decode('utf-8', 'replace'))
    if '\n'.join(lines).strip() != '':
        yield '\n'.join(lines)


def frame_data_payloads(frame: str) -> list[str]:
    """The `data:` payloads of one SSE frame, in order, without the `data:` prefix."""
    payloads: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if trimmed.startswith('data:'):
            payloads.append(trimmed[5:].strip())
    return payloads


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
        if is_obj(chunk):
            rewritten = {**chunk, 'model': model}
            lines.append(f'data: {json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def _chunk_is_finish(chunk: Json) -> bool:
    """True when any choice of the chunk carries a finish_reason: the chunk ends the answer."""
    choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
    return any(is_obj(choice) and choice.get('finish_reason') is not None for choice in choices)


def _chunk_is_usage_only(chunk: Json) -> bool:
    """True for the trailing usage chunk: a usage object and no choices at all."""
    choices = chunk.get('choices')
    return is_obj(chunk.get('usage')) and isinstance(choices, list) and len(choices) == 0


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream normalized for the wire, every payload's `model` the dialled id.

    The upstream's own [DONE] is held back and re-emitted once at the end: when
    the stream carried no usage chunk, one is appended first (empty choices, the
    estimated usage), because the wire check reads usage from any chunk and a
    stream without it fails the usage check outright. A stream cut short without
    [DONE] is still closed properly.

    The order is normalized on the way through, because a provider's stream is
    not guaranteed to arrive in the order the wire checks demand: the first
    finish chunk is the only one forwarded, a content delta or a second finish
    after it is dropped, and after the finish only the empty-choices usage
    chunk may still pass. What survives the filter is exactly one finish
    chunk, then usage, then the entry's own [DONE].
    """
    usage_seen = False
    text_chars = 0
    finish_seen = False
    for frame in sse_frames(chunks):
        payloads = frame_data_payloads(frame)
        if payloads == ['[DONE]']:
            continue
        if not payloads:
            # A frame with no data line (a comment, an event line) carries no
            # answer; after the finish nothing but usage may pass at all.
            if not finish_seen:
                yield frame_with_model(frame, model)
            continue
        for payload in payloads:
            if payload == '' or payload == '[DONE]':
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                chunk = None
            if not is_obj(chunk):
                if not finish_seen:
                    yield frame_with_model(f'data: {payload}', model)
                continue
            if finish_seen and not _chunk_is_usage_only(chunk):
                # Past the finish, only the usage chunk may still pass: a late
                # content delta or a second finish fails the wire checks.
                continue
            if is_obj(chunk.get('usage')):
                usage_seen = True
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            if choices and is_obj(choices[0]):
                delta = choices[0].get('delta') if is_obj(choices[0].get('delta')) else {}
                if isinstance(delta.get('content'), str):
                    text_chars += len(delta['content'])
            if _chunk_is_finish(chunk):
                finish_seen = True
            yield frame_with_model(f'data: {payload}', model)
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


class AnthropicStreamer:
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text
    block or tool call, then message_delta with the stop reason and usage, then
    message_stop. `model` is the id the entry dialled, named back to the client.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.started = False
        self.finished = False
        # The stop reason a finish_reason named, held until the stream's usage chunk has arrived.
        self.pending_stop: str | None = None
        self.text_index = -1
        self.next_index = 0
        self.tool_indexes: dict[object, int] = {}
        self.input_tokens: int | float = 0
        self.output_tokens: int | float = 0
        self.text_chars = 0

    def run(self, chunks):
        """The Anthropic events for one upstream stream, in order."""
        for frame in sse_frames(chunks):
            yield from self.frame_events(frame)
        if not self.finished:
            yield from self.start_events()
            yield from self.finish_events(self.pending_stop if self.pending_stop is not None else 'end_turn')

    # -- event builders -------------------------------------------------------

    def _block_start(self, index: int, block: Json) -> str:
        return sse_event('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': block})

    def _block_delta(self, index: int, delta: Json) -> str:
        return sse_event('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': delta})

    def _block_stop(self, index: int) -> str:
        return sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index})

    def start_events(self) -> list[str]:
        """The message_start event, once, on the first chunk that carries a choice."""
        if self.started:
            return []
        self.started = True
        message: Json = {
            'id': 'msg_stream',
            'type': 'message',
            'role': 'assistant',
            'model': self.model,
            'content': [],
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {'input_tokens': self.input_tokens, 'output_tokens': 0},
        }
        return [sse_event('message_start', {'type': 'message_start', 'message': message})]

    def _close_blocks(self) -> list[str]:
        """content_block_stop for every block still open, text first, then tools in index order."""
        out: list[str] = []
        if self.text_index != -1:
            out.append(self._block_stop(self.text_index))
            self.text_index = -1
        for index in sorted(self.tool_indexes.values()):
            out.append(self._block_stop(index))
        self.tool_indexes.clear()
        return out

    def finish_events(self, stop_reason: str) -> list[str]:
        """message_delta (stop reason + usage) and message_stop, once."""
        if self.finished:
            return []
        self.finished = True
        # The wire check needs an output count of at least 1: the provider's own
        # count when it sent one, else an estimate from the streamed text.
        output = self.output_tokens if self.output_tokens > 0 else max(1, self.text_chars // 4)
        delta: Json = {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': {'output_tokens': output},
        }
        return [*self._close_blocks(), sse_event('message_delta', delta), sse_event('message_stop', {'type': 'message_stop'})]

    # -- per-frame translation --------------------------------------------------

    def frame_events(self, frame: str) -> list[str]:
        """The Anthropic events one OpenAI frame becomes."""
        out: list[str] = []
        for payload in frame_data_payloads(frame):
            if payload == '[DONE]':
                out.extend(self.start_events())
                out.extend(self.finish_events(self.pending_stop if self.pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_obj(chunk):
                continue
            if is_obj(chunk.get('usage')):
                self.input_tokens = as_num_or_zero(chunk['usage'].get('prompt_tokens'))
                self.output_tokens = as_num_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_obj(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_obj(choice.get('delta')) else {}
            out.extend(self.start_events())
            out.extend(self._text_events(delta))
            out.extend(self._tool_events(delta))
            # Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
            # OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
            # output tokens.
            if choice.get('finish_reason') is not None:
                self.pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    def _text_events(self, delta: Json) -> list[str]:
        """content_block_start/delta for a text delta, opening the block on first text."""
        content = delta.get('content')
        if not isinstance(content, str) or content == '':
            return []
        self.text_chars += len(content)
        out: list[str] = []
        if self.text_index == -1:
            self.text_index = self.next_index
            self.next_index += 1
            out.append(self._block_start(self.text_index, {'type': 'text', 'text': ''}))
        out.append(self._block_delta(self.text_index, {'type': 'text_delta', 'text': content}))
        return out

    def _tool_events(self, delta: Json) -> list[str]:
        """content_block_start/delta for tool-call deltas, opening a block per upstream index."""
        calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
        out: list[str] = []
        for call in calls:
            if not is_obj(call):
                continue
            upstream_index = call['index'] if is_num(call.get('index')) else len(self.tool_indexes)
            index = self.tool_indexes.get(upstream_index)
            if index is None:
                if self.text_index != -1:
                    out.append(self._block_stop(self.text_index))
                    self.text_index = -1
                index = self.next_index
                self.next_index += 1
                self.tool_indexes[upstream_index] = index
                fn = call.get('function') if is_obj(call.get('function')) else {}
                call_id = call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}'
                block: Json = {
                    'type': 'tool_use',
                    'id': call_id,
                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                    'input': {},
                }
                out.append(self._block_start(index, block))
            fn = call.get('function') if is_obj(call.get('function')) else {}
            if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                out.append(self._block_delta(index, {'type': 'input_json_delta', 'partial_json': fn['arguments']}))
        return out


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence (see AnthropicStreamer)."""
    yield from AnthropicStreamer(model).run(chunks)


# --- the trial state ---------------------------------------------------------

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
    base = GATEWAY_URL if GATEWAY_URL.endswith('/v1') else GATEWAY_URL + '/v1'
    return base + '/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    listing: list[Json] = []
    for m in all_models(BRIEF):
        listing.append({'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']})
    return listing


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
    eligible = [m for m in all_models(BRIEF) if meets_requirements(requires, m)]
    with cooled_lock:
        cooled = set(cooled_down)

    def rank(model: Json) -> tuple[bool, int | float]:
        return (model['id'] in cooled, -as_num_or_zero(model['rating_milli']))

    return sorted(eligible, key=rank)


def _body_has_image(body: Json) -> bool:
    """True when any message carries an image part, which needs a vision model."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_obj(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_obj(block) and block.get('type') in ('image', 'image_url'):
                return True
    return False


def _body_requirements(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_obj(response_format) else None
    return {
        'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
        'vision': _body_has_image(body),
        'json': response_type in ('json_object', 'json_schema'),
    }


def resolve_requirements(body: Json) -> Json:
    """The requirements of the prompt now arriving, from the manifest and the body together.

    The brief's row by arrival order is the declared source; the body is the
    check against a manifest row that understates the request (tools declared,
    a JSON response_format, image parts). A requirement either source names
    must be met, so the stricter of the two wins.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    manifest_req: Json = prompts[index]['requires'] if index < len(prompts) else {}
    body_req = _body_requirements(body)
    merged: Json = {}
    for key in ('tools', 'vision', 'json'):
        if manifest_req.get(key) is True or body_req.get(key) is True:
            merged[key] = True
    min_context = manifest_req.get('min_context')
    if is_num(min_context):
        merged['min_context'] = min_context
    return merged


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
        yield from iter(lambda: reader(65536), b'')

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
        response = urllib.request.urlopen(request, timeout=DIAL_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


# --- failover (max 2 recovery steps per prompt, plus the 401 candidate chain) ---

def _output_cap(body: Json) -> int:
    """The output cap the gateway must receive: the client's, clamped to 8192, else 512."""
    cap = body.get('max_tokens')
    if not is_num(cap):
        cap = body.get('max_completion_tokens')
    if not is_num(cap) or cap <= 0:
        return DEFAULT_UPSTREAM_TOKENS
    return min(int(cap), MAX_UPSTREAM_TOKENS)


def upstream_body(body: Json) -> Json:
    """The body as the gateway must receive it: an output cap is always named.

    The gateway refuses a request with no max_tokens, or one over 8192. The
    client always sends one in practice; when it does not, 512 is the default,
    and a cap over 8192 is clamped rather than refused. A streaming request asks
    for its usage chunk, which the wire checks read.
    """
    out = dict(body)
    out['max_tokens'] = _output_cap(body)
    out.pop('max_completion_tokens', None)
    if body.get('stream') is True:
        out['stream_options'] = {'include_usage': True}
    return out


def _code_text(code: object) -> str:
    """An error code as short text: a string as-is, a number stringified, anything else ''."""
    if isinstance(code, str):
        return code[:64]
    if isinstance(code, (int, float)) and not isinstance(code, bool):
        return str(int(code))[:64]
    return ''


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
    if not is_obj(value):
        return ''
    error = value.get('error')
    if not is_obj(error):
        return ''
    code = _code_text(error.get('code'))
    if code != '':
        return code
    err_type = error.get('type')
    return err_type[:64] if isinstance(err_type, str) and err_type != '' else ''


def _record_refusal(failures: list[Json], status: int, code: str, model_id: str) -> None:
    """One refusal on the trail, and the model cooled down for later prompts."""
    failures.append({'status': status, 'code': code, 'model': model_id})
    mark_cooled(model_id)


def _single_dial(model_id: str, prepared: Json, failures: list[Json]) -> tuple[Upstream | None, str]:
    """One gateway call. Returns (upstream, outcome) with outcome 'ok', 'refused',
    'timeout', or 'unreachable'; every refusal is recorded on `failures`."""
    try:
        upstream = dial(model_id, prepared)
    except TimeoutError:
        # A hung route: record it as a 504 and let the caller fail over at once —
        # retrying the same hung route would only burn the rest of the prompt's 30 s window.
        _record_refusal(failures, 504, 'timeout', model_id)
        return None, 'timeout'
    except Exception:
        _record_refusal(failures, 502, 'upstream_unreachable', model_id)
        return None, 'unreachable'
    if upstream.ok:
        return upstream, 'ok'
    _record_refusal(failures, upstream.status, error_code_of(upstream.text()), model_id)
    return upstream, 'refused'


def dial_with_recovery(requires: Json, body: Json) -> tuple[Json | None, Upstream | None, list[Json]]:
    """Dial the best model for `requires`, recovering from refusals within the budget.

    At most 2 recovery steps per prompt: one retry on the same model for a
    429/5xx, then one failover to the next-best model meeting the requirements.
    A 401 is the exception: it refuses the model, not the prompt — the key may
    lack that one model while the next candidate still answers — so a 401
    fails over to the next candidate and the prompt is given up only when
    every candidate has refused (a key dead for all models still fails the
    prompt, after each candidate has been tried once). Every refused model is
    cooled down for later prompts. Returns (model, upstream,
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
        upstream, outcome = _single_dial(attempt['id'], prepared, failures)
        if outcome == 'ok':
            return attempt, upstream, failures
        if outcome == 'refused' and upstream.status == 401:
            # This model refused the key; the next candidate may still take
            # it. Chain through the untried candidates, best first.
            rest = [m for m in ranked if m['id'] not in tried]
            if rest:
                attempt = rest[0]
                continue
            return None, None, failures
        if outcome == 'refused':
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
            elif self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
            elif self.command == 'POST' and path == '/v1/messages':
                self.messages()
            else:
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
        raw = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
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
        try:
            size = int(length) if length is not None else -1
        except ValueError:
            size = -1
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
        return value if is_obj(value) else None

    def refused(self, wire: str, status: int, text: str) -> None:
        """A refused or broken gateway call as a 502 in the client's wire: the prompt is lost."""
        suffix = f': {text[:500]}' if text != '' else ''
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway answered HTTP {status}{suffix}')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway answered HTTP {status}{suffix}')

    # -- the routes ---------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            payload = anthropic_model_list(listing)
        else:
            payload = openai_model_list(listing)
        self.send_json(200, payload)

    def trail_headers(self, failures: list[Json]) -> dict[str, str]:
        """The failures trail header, when a prompt was served after refusals.

        The env reads at most 8 rows and 8 KiB; a served prompt's refusals are the
        only way the failover column can see them, so they ride on the answer.
        """
        if not failures:
            return {}
        trail = json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)
        return {'x-makefun-upstream-failures': trail}

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        self.serve('openai', body)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        self.serve('anthropic', body)

    def serve(self, wire: str, body: Json) -> None:
        """One completion request on either wire: resolve, translate, dial, render.

        Anthropic requests are translated to Chat Completions before the dial
        and translated back for the answer; OpenAI requests pass through. The
        dial, the recovery, and the trail header are the same path for both.
        """
        requires = resolve_requirements(body)
        if not candidates_for(requires):
            error_type = 'overloaded_error' if wire == 'anthropic' else 'no_free_model'
            self.send_error_body(wire, 503, error_type, 'no free model in the brief meets this request')
            return
        if wire == 'anthropic':
            dial_body = anthropic_to_openai(body, 'auto')
        else:
            dial_body = {**body, 'model': 'auto'}
        try:
            route, upstream, failures = dial_with_recovery(requires, dial_body)
        except Exception as err:  # an unreachable gateway is a 502 to this prompt
            error_type = 'api_error' if wire == 'anthropic' else 'upstream_unreachable'
            self.send_error_body(wire, 502, error_type, str(err))
            return
        if route is None or upstream is None:
            self.refused(wire, 502, 'the gateway refused every model that meets this request')
            return
        trail = self.trail_headers(failures)
        if body.get('stream') is True:
            self.stream_headers(trail)
            try:
                if wire == 'anthropic':
                    events = anthropic_events_from_openai(upstream.chunks(), route['id'])
                else:
                    events = openai_stream_with_model(upstream.chunks(), route['id'])
                for event in events:
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_obj(parsed):
            self.refused(wire, 502, 'the gateway answered a body that is not JSON')
            return
        parsed['usage'] = whole_usage(parsed)
        if wire == 'anthropic':
            self.send_json(200, openai_to_anthropic(parsed, route['id']), trail)
        else:
            self.send_json(200, openai_json_with_model(parsed, route['id']), trail)


# --- register and poll -----------------------------------------------------------

def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the trial's prompt count."""
    payload = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=payload.encode('utf-8'),
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
    if is_obj(reply) and is_num(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def _work_is_done() -> bool:
    """One GET /work: True only when the env answers done; False on any hiccup."""
    try:
        with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
            if response.status != 200:
                return False
            work = json.loads(response.read().decode('utf-8', 'replace'))
    except Exception:  # the env is not reachable yet: try again on the next tick
        return False
    return is_obj(work) and work.get('done') is True


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        if _work_is_done():
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
