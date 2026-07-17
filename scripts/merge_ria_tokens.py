#!/usr/bin/env python3
"""Merge ruiN + yaN → ria in tokens.jsonl files. Usage: python merge_ria_tokens.py <file1> <file2> ..."""
import json, re, sys

for path in sys.argv[1:]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        continue
    entries = [json.loads(l) for l in lines if l.strip()]
    if len(entries) < 2:
        continue
    new, i, changed = [], 0, False
    while i < len(entries):
        w = entries[i]['word']
        if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
                and re.match(r'^ya[0-5]$', entries[i + 1]['word'])):
            a, b = entries[i], entries[i + 1]
            new.append({'word': 'ria', 'start_ms': a['start_ms'], 'end_ms': b['end_ms'],
                        'start_s': a['start_s'], 'end_s': b['end_s'],
                        'type': a.get('type', 'word')})
            i += 2; changed = True
        else:
            new.append(entries[i]); i += 1
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            for e in new:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
