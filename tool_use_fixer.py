import os
import re
import uuid

SMART_DOUBLE_QUOTES = frozenset({
    '\u00ab', '\u201c', '\u201d', '\u275e',
    '\u201f', '\u201e', '\u275d', '\u00bb',
})

SMART_SINGLE_QUOTES = frozenset({
    '\u2018', '\u2019', '\u201a', '\u201b',
})


def normalize_tool_arguments(args):
    """字段映射：file_path → path"""
    if not isinstance(args, dict):
        return args
    if 'file_path' in args and 'path' not in args:
        args['path'] = args.pop('file_path')
    return args


def _build_fuzzy_pattern(text):
    """构建容错正则：引号互换、空白差异、反斜杠差异"""
    pattern_parts = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in SMART_DOUBLE_QUOTES or ch == '"':
            pattern_parts.append('["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]')
        elif ch in SMART_SINGLE_QUOTES or ch == "'":
            pattern_parts.append("['\u2018\u2019\u201a\u201b]")
        elif ch in (' ', '\t'):
            pattern_parts.append(r'\s+')
        elif ch == '\\':
            pattern_parts.append(r'\\{1,2}')
        else:
            pattern_parts.append(re.escape(ch))
        i += 1
    return ''.join(pattern_parts)


def _replace_smart_quotes(text):
    """将智能引号替换为普通引号"""
    result = list(text)
    for i, ch in enumerate(result):
        if ch in SMART_DOUBLE_QUOTES:
            result[i] = '"'
        elif ch in SMART_SINGLE_QUOTES:
            result[i] = "'"
    return ''.join(result)


def repair_exact_match_tool_arguments(tool_name, args):
    """修复 StrReplace / search_replace 工具的 old_string 精确匹配问题"""
    if not isinstance(args, dict):
        return args

    lower_name = (tool_name or '').lower()
    if 'str_replace' not in lower_name and 'search_replace' not in lower_name:
        return args

    old_string = args.get('old_string') or args.get('old_str')
    if not old_string:
        return args

    file_path = args.get('path') or args.get('file_path')
    if not file_path or not os.path.isfile(file_path):
        return args

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return args

    # 已经精确匹配，无需修复
    if old_string in content:
        return args

    # 构建容错正则尝试匹配
    pattern = _build_fuzzy_pattern(old_string)
    try:
        matches = list(re.finditer(pattern, content))
    except re.error:
        return args

    # 仅在唯一匹配时修复
    if len(matches) != 1:
        return args

    matched_text = matches[0].group()

    # 更新 old_string
    if 'old_string' in args:
        args['old_string'] = matched_text
    elif 'old_str' in args:
        args['old_str'] = matched_text

    # 同步修复 new_string 中的引号
    new_string = args.get('new_string') or args.get('new_str')
    if new_string:
        fixed_new = _replace_smart_quotes(new_string)
        if 'new_string' in args:
            args['new_string'] = fixed_new
        elif 'new_str' in args:
            args['new_str'] = fixed_new

    return args


def fix_tool_use_response(response_data):
    """修复响应中的 tool_use 问题"""
    if not isinstance(response_data, dict):
        return response_data

    content = response_data.get('content', [])
    if not isinstance(content, list):
        return response_data

    has_tool_use = False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'tool_use':
            has_tool_use = True
            # 缺少 id 则生成临时 id
            if not block.get('id'):
                block['id'] = f'toolu_{uuid.uuid4().hex[:24]}'

    # 有 tool_use 块但 stop_reason 不对，修正
    if has_tool_use and response_data.get('stop_reason') != 'tool_use':
        response_data['stop_reason'] = 'tool_use'

    return response_data
