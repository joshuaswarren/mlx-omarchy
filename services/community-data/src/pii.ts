// Server-side PII rejection scan: defense in depth behind the
// collector's local redaction. Scans the JSON-serialized summary text;
// on any hit the submission is REFUSED, never stored. Patterns mirror
// the collector's Redactor so a string the collector would have
// redacted is also refused server-side if redaction was bypassed.
//
// Matches are reported as kind counts only; matched text is never
// echoed back, so the error response cannot leak the PII itself.

export type PiiKinds = Record<string, number>;

const PII_RE = new RegExp(
  [
    // MAC address (colon-separated).
    "(?<mac>\\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\\b)",
    // IPv6: link-local / ULA prefixes, then the full 8-group form.
    "(?<ipv6>\\b(?:fe80|fd[0-9a-f]{2}|fc[0-9a-f]{2})(?::[0-9a-fA-F]{0,4}){1,7}(?:%\\w+)?\\b|\\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\\b)",
    // IPv4, optional CIDR suffix.
    "(?<ipv4>\\b\\d{1,3}(?:\\.\\d{1,3}){3}(?:/\\d{1,3})?\\b)",
    // UUID.
    "(?<uuid>\\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\b)",
    // Serial numbers in JSON/text shapes; skips bare null/true/false.
    "(?<serial>\\bserial(?:[_-]?number)?\\b[\"'\\s:=]{1,4}(?!null\\b|true\\b|false\\b)[^\"',\\s}{]{2,})",
    // Home paths: POSIX and Windows (backslashes arrive JSON-escaped).
    "(?<home_path>/(?:home|Users)/[A-Za-z0-9._-]{1,64}|[A-Za-z]:\\\\+Users\\\\+[A-Za-z0-9._-]{1,64})",
    // Credential shapes: token prefixes, AWS keys, Slack tokens, JWTs,
    // bearer headers, and key=value assignments with quoted secrets.
    "(?<credential>\\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\\b|\\bgithub_pat_[A-Za-z0-9_]{20,}\\b|\\bAKIA[0-9A-Z]{16}\\b|\\bxox[baprs]-[A-Za-z0-9-]{10,}\\b|\\bsk-[A-Za-z0-9_-]{20,}\\b|\\beyJ[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{5,}\\b|\\bBearer\\s+[A-Za-z0-9._~+/=-]{8,}|(?:password|passwd|secret|api_?key|private_?key|auth_?token)\\w*\\\\*[\"']?\\s*[:=]\\s*\\\\*[\"'][^\"']{4,}[\"'])",
    // Hostname shapes: mDNS and common private suffixes.
    "(?<hostname>\\.(?:local|lan|home|internal)\\b)",
  ].join("|"),
  "i",
);
const MAX_REPORTED_HITS = 50;

export function scanPii(text: string): PiiKinds | null {
  const kinds: PiiKinds = {};
  const re = new RegExp(PII_RE.source, PII_RE.flags + "g");
  let match: RegExpExecArray | null;
  let hits = 0;
  while ((match = re.exec(text)) !== null) {
    const groups = match.groups ?? {};
    const kind = Object.keys(groups).find((k) => groups[k] !== undefined);
    if (kind === undefined) continue;
    kinds[kind] = (kinds[kind] ?? 0) + 1;
    hits++;
    if (hits >= MAX_REPORTED_HITS) break;
    if (match[0].length === 0) re.lastIndex++;
  }
  return hits > 0 ? kinds : null;
}
