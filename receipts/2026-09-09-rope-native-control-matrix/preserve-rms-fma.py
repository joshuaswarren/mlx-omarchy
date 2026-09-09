import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
words = list(struct.unpack(f'<{len(raw) // 4}I', raw))
assert words[0] == 0x07230203
instructions = []
pos = 5
while pos < len(words):
    count = words[pos] >> 16
    assert count > 0 and pos + count <= len(words)
    instructions.append(words[pos:pos + count])
    pos += count
imports = set()
for inst in instructions:
    if inst[0] & 65535 == 11:
        name = struct.pack(f'<{len(inst) - 2}I', *inst[2:]).rstrip(b'\0')
        if name == b'GLSL.std.450':
            imports.add(inst[1])
results = {i[2] for i in instructions if i[0] & 65535 == 12 and i[3] in imports and i[4] == 50}
assert results, path
existing = {i[1] for i in instructions if i[0] & 65535 == 71 and i[2] == 42}
insert_at = next(n for n, i in enumerate(instructions) if i[0] & 65535 == 71)
annotations = [[(3 << 16) | 71, result, 42] for result in sorted(results - existing)]
instructions[insert_at:insert_at] = annotations
output = words[:5] + [word for inst in instructions for word in inst]
path.write_bytes(struct.pack(f'<{len(output)}I', *output))
print(f'{path.name}: protected {len(results)} FMA results')
