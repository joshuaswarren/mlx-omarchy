"""Verify the pinned mirror against its original LibriSpeech archive member."""

import hashlib
import json
import tarfile
import time
import urllib.request

MIRROR = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/8d141c84e3f84c54cd7bbaa851d24edd0f559734/1.flac"
ARCHIVE = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"
MEMBER = "LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
SHA256 = "30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2"

with urllib.request.urlopen(MIRROR, timeout=20) as response:
    candidate = response.read(183319)
assert len(candidate) == 183318
assert hashlib.sha256(candidate).hexdigest() == SHA256
started = time.monotonic()
with (
    urllib.request.urlopen(ARCHIVE, timeout=20) as response,
    tarfile.open(fileobj=response, mode="r|gz") as archive,
):
    for member in archive:
        if time.monotonic() - started > 180:
            raise TimeoutError("primary archive scan exceeded 180 seconds")
        if member.name != MEMBER:
            continue
        assert member.isfile() and member.size == len(candidate)
        stream = archive.extractfile(member)
        assert stream is not None and stream.read() == candidate
        print(json.dumps({"canonical_archive": ARCHIVE, "canonical_member": MEMBER,
                          "mirror": MIRROR, "sha256": SHA256,
                          "bytes": len(candidate), "byte_identical": True}))
        break
    else:
        raise RuntimeError("primary archive does not contain the pinned member")
