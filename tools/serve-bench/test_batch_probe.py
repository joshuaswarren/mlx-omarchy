#!/usr/bin/env python3
"""CPU-only check of capture bookkeeping, not model qualification."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

source = ast.parse(Path(__file__).with_name('server_ids_probe.py').read_text())
namespace = {'json': json}
functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
             and n.name in ('sha16', 'make_batch_probe')]
exec(compile(ast.Module(body=functions, type_ignores=[]), '<probe>', 'exec'), namespace)
assert 'make_batch_probe' in namespace, 'batch path has no completion capture'
events = []
a, b = NS(), NS()
response = lambda uid, token, finish=None: NS(uid=uid, token=token, finish_reason=finish)
next_probe = namespace['make_batch_probe'](lambda self: self.result, events.append)
a.result = ([], [response(1, 10), response(2, 20)])
assert next_probe(a) is a.result
assert not events
b.result = ([], [response(1, 99, 'length')])
next_probe(b)
a.result = ([], [response(1, 11, 'length'), response(2, 21, 'stop')])
next_probe(a)
assert [e['ids'] for e in events] == [[99], [10, 11], [20, 21]]
a.result = ([], [])
next_probe(a)
assert len(events) == 3
assert not a._probe_token_ids and not b._probe_token_ids
print('PASS batch finish capture: final request, interleaved UIDs, instance isolation')
