"""
Security & privacy tools: password_generator, hash_tool, secret_scanner, pii_redactor, encryption_tool.

* password_generator uses the `secrets` module; strength is entropy math (length * log2(pool)), not a guess.
* hash_tool: sha2/sha3/blake2 only (md5/sha1 are refused); HMAC key comes from an env var, verification uses
  hmac.compare_digest.
* secret_scanner: regex + Shannon entropy; findings are always redacted (never the full secret). Unlike other file
  tools it may open secret-named files such as .env *because* finding leaked secrets is its job; it still stays
  inside the workspace and never prints the value.
* pii_redactor: deterministic regex redaction (email, Indian phone, Aadhaar-format, PAN, Luhn-valid cards).
* encryption_tool needs `cryptography`. Secret: `password` argument (unavoidable for an LLM tool, never logged/returned)
  or env AGENT_ENCRYPTION_KEY; passphrases are stretched with scrypt + random salt.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import os
import re
import secrets
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import tool

from ._common import (
    ToolError,
    atomic_write_bytes,
    atomic_write_text,
    guard,
    ok_json,
    read_text_file,
    require,
    safe_output_path,
    safe_path,
    workspace_root,
)

MAX_FILE_HASH_BYTES = 2_000_000_000
MAX_CRYPT_BYTES = 50_000_000
MAX_SCAN_FILE_BYTES = 1_000_000
MAX_SCAN_FILES = 2000
MAX_TEXT_CHARS = 2_000_000


# ---------------------------------------------------------------------------
# password_generator
# ---------------------------------------------------------------------------
_AMBIGUOUS = set("Il1O0o")
_SYMBOLS = "!@#$%^&*()-_=+[]{};:,.?"


def _strength_label(bits: float) -> str:
    if bits < 40:
        return "weak"
    if bits < 60:
        return "fair"
    if bits < 80:
        return "good"
    return "strong"


def _wordlist() -> List[str]:
    """1024 distinct lowercase words generated from syllables (consonant+vowel+consonant+vowel...), deterministic."""
    cons, vows = "bdfgklmnprstvz", "aeiou"
    syl = [c + v for c in cons for v in vows]  # 70 syllables
    words = []
    for a in syl:
        for b in syl:
            words.append(a + b)  # 4900 pronounceable 4-letter words
    return words[:: len(words) // 1024][:1024]


@tool
@guard
def password_generator(length: int = 16, count: int = 1, use_symbols: bool = True, use_digits: bool = True,
                       use_uppercase: bool = True, exclude_ambiguous: bool = False, passphrase: bool = False,
                       words: int = 5) -> str:
    """Generate cryptographically secure random passwords (secrets module) with an entropy-based strength estimate.

    Args:
        length: password length, 8-128 (default 16).
        count: how many passwords to generate, 1-10 (default 1).
        use_symbols: include punctuation symbols (default True).
        use_digits: include digits (default True).
        use_uppercase: include uppercase letters (default True).
        exclude_ambiguous: drop look-alike characters such as l, 1, I, O, 0 (default False).
        passphrase: generate a word passphrase instead (words joined by "-", from a built-in 1024-word list).
        words: passphrase word count, 4-12 (default 5); each word adds exactly 10 bits.
    """
    count = int(count)
    if passphrase:
        words = int(words)
        if not 4 <= words <= 12 or not 1 <= count <= 10:
            raise ToolError("words must be 4-12 and count 1-10")
        wl = _wordlist()
        phrases = ["-".join(secrets.choice(wl) for _ in range(words)) for _ in range(count)]
        bits = words * math.log2(len(wl))
        return ok_json({"passphrases": phrases, "words": words, "wordlist_size": len(wl), "entropy_bits": round(bits, 1),
                        "strength": _strength_label(bits),
                        "note": "entropy = words x log2(wordlist size); the wordlist is public so this is exact for random picks."})
    length = int(length)
    if not 8 <= length <= 128:
        raise ToolError("length must be between 8 and 128")
    if not 1 <= count <= 10:
        raise ToolError("count must be between 1 and 10")
    groups = [string.ascii_lowercase]
    if use_uppercase:
        groups.append(string.ascii_uppercase)
    if use_digits:
        groups.append(string.digits)
    if use_symbols:
        groups.append(_SYMBOLS)
    if exclude_ambiguous:
        groups = ["".join(c for c in g if c not in _AMBIGUOUS) for g in groups]
    pool = "".join(groups)
    rng = secrets.SystemRandom()
    out = []
    for _ in range(count):
        chars = [secrets.choice(g) for g in groups]  # guarantee one of each enabled class
        chars += [secrets.choice(pool) for _ in range(length - len(chars))]
        rng.shuffle(chars)
        out.append("".join(chars))
    # Upper bound on entropy (uniform draws from the pool); the one-per-class guarantee slightly lowers it.
    bits = length * math.log2(len(pool))
    return ok_json({
        "passwords": out,
        "length": length,
        "pool_size": len(pool),
        "entropy_bits": round(bits, 1),
        "strength": _strength_label(bits),
        "note": "entropy = length x log2(pool size), assumes the attacker knows the rules. "
                "Store passwords in a password manager; they are shown only in this reply.",
    })


# ---------------------------------------------------------------------------
# hash_tool
# ---------------------------------------------------------------------------
_INSECURE = {"md5", "sha1"}
_ALGOS = {"md5": "md5", "sha1": "sha1", "sha256": "sha256", "sha384": "sha384", "sha512": "sha512", "sha3_256": "sha3_256",
          "sha3_512": "sha3_512", "blake2b": "blake2b", "blake2s": "blake2s"}


def _algo(name: str) -> str:
    n = (name or "sha256").strip().lower().replace("-", "")
    n = {"sha3256": "sha3_256", "sha3512": "sha3_512"}.get(n, n)
    if n not in _ALGOS:
        raise ToolError(f"unsupported algorithm '{name}'; use one of: {', '.join(_ALGOS)}")
    return n


def _flag(algo: str, out: dict) -> dict:
    if algo in _INSECURE:
        out["insecure"] = True
        out["warning"] = f"{algo} is cryptographically broken: fine for checksums of non-hostile data, never for security"
    return out


def _digest_input(text: str, path: str, algo: str, key: Optional[bytes] = None) -> str:
    def new():
        return hmac.new(key, digestmod=algo) if key is not None else hashlib.new(algo)

    h = new()
    if path and path.strip():
        p = safe_path(path, must_exist=True)
        if not p.is_file():
            raise ToolError("path is not a file")
        if p.stat().st_size > MAX_FILE_HASH_BYTES:
            raise ToolError("file is too large to hash (limit 2 GB)")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        if text is None or text == "":
            raise ToolError("provide `text` or a workspace file `path`")
        if len(text) > MAX_TEXT_CHARS:
            raise ToolError("text too long")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def _hmac_key(key_env: str) -> bytes:
    name = (key_env or "AGENT_HMAC_KEY").strip()
    if not re.fullmatch(r"AGENT_HMAC_[A-Z0-9_]*|AGENT_HMAC", name):
        raise ToolError("key_env must be AGENT_HMAC_KEY or an env var name starting with AGENT_HMAC_")
    val = os.environ.get(name, "")
    if not val:
        raise ToolError(f"environment variable {name} is not set; set it to the shared HMAC secret first")
    return val.encode("utf-8")


@tool
@guard
def hash_tool(action: str = "hash", text: str = "", path: str = "", algorithm: str = "sha256",
              expected: str = "", key_env: str = "AGENT_HMAC_KEY") -> str:
    """Compute or verify cryptographic hashes and HMACs of text or a workspace file (sha256/384/512, sha3, blake2; md5/sha1 allowed only with an insecure warning).

    Args:
        action: "hash" (digest), "verify" (compare digest with `expected`), "hmac" (keyed digest), "hmac_verify" (check `expected` HMAC in constant time).
        text: text to process (use this or `path`).
        path: workspace file to process (streamed, up to 2 GB).
        algorithm: sha256 (default), sha384, sha512, sha3_256, sha3_512, blake2b, blake2s; md5/sha1 work but are flagged insecure.
        expected: hex digest to compare against for the verify actions.
        key_env: for hmac actions, the env var holding the secret (default AGENT_HMAC_KEY; name must start with AGENT_HMAC). The key is never accepted as an argument or returned.
    """
    a = (action or "hash").strip().lower()
    algo = _algo(algorithm)
    if a == "hash":
        return ok_json(_flag(algo, {"algorithm": algo, "digest": _digest_input(text, path, algo)}))
    if a == "verify":
        exp = (expected or "").strip().lower()
        if not exp:
            raise ToolError("`expected` is required for verify")
        got = _digest_input(text, path, algo)
        return ok_json(_flag(algo, {"algorithm": algo, "match": hmac.compare_digest(got.encode(), exp.encode())}))
    if a in ("hmac", "hmac_verify"):
        key = _hmac_key(key_env)
        got = _digest_input(text, path, algo, key)
        if a == "hmac":
            return ok_json(_flag(algo, {"algorithm": f"hmac-{algo}", "digest": got}))
        exp = (expected or "").strip().lower()
        if not exp:
            raise ToolError("`expected` is required for hmac_verify")
        return ok_json(_flag(algo, {"algorithm": f"hmac-{algo}", "match": hmac.compare_digest(got.encode(), exp.encode())}))
    raise ToolError("action must be one of: hash, verify, hmac, hmac_verify")


# ---------------------------------------------------------------------------
# secret_scanner
# ---------------------------------------------------------------------------
_SECRET_RULES = [
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("password_in_url", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:([^\s/@]{4,})@")),
    ("hardcoded_credential", re.compile(
        r"(?i)\b[\w.\-]*(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key)[\w.\-]*\s*[:=]\s*[\"']?([^\s\"'#,;]{8,})")),
]
_HIGH_ENTROPY_RE = re.compile(r"[A-Za-z0-9+/_\-=]{28,}")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}
_PLACEHOLDER_RE = re.compile(r"(?i)^(?:x+|\*+|<.*>|\$\{.*\}|\{\{.*\}\}|your[_\-].*|changeme|example.*|placeholder.*|none|null|true|false|process\.env.*|os\.(?:environ|getenv).*)$")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _mask(s: str) -> str:
    """Never reveal more than 4 leading characters; length is shown so the finding can be located."""
    return (s[:4] + "***" if len(s) > 8 else "***") + f" (len {len(s)})"


def _scan_text(text: str, label: str, limit: int, min_entropy: float) -> List[dict]:
    findings: List[dict] = []
    for ln, line in enumerate(text.splitlines(), 1):
        if len(line) > 5000:
            line = line[:5000]
        seen_spans: List[tuple] = []
        for rule, rx in _SECRET_RULES:
            for m in rx.finditer(line):
                secret = m.group(1) if rx.groups else m.group(0)
                if rule == "hardcoded_credential" and (_PLACEHOLDER_RE.match(secret) or secret.isalpha() and _entropy(secret) < 2.5):
                    continue
                if rule == "password_in_url" and _PLACEHOLDER_RE.match(secret):
                    continue
                seen_spans.append(m.span())
                findings.append({"file": label, "line": ln, "type": rule, "preview": _mask(secret)})
        for m in _HIGH_ENTROPY_RE.finditer(line):
            tok = m.group(0)
            if any(a <= m.start() < b or a < m.end() <= b for a, b in seen_spans):
                continue
            if not (any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)):
                continue
            if re.fullmatch(r"[0-9a-fA-F\-]{32,36}", tok) and tok.count("-") in (0, 4) and len(tok) in (32, 36):
                continue  # UUID / 128-bit hex id: high entropy but not by itself a secret
            if "/" in tok and tok.count("/") > 2:
                continue
            if _entropy(tok) >= min_entropy:
                findings.append({"file": label, "line": ln, "type": "high_entropy_string", "preview": _mask(tok),
                                 "entropy_bits_per_char": round(_entropy(tok), 2)})
        if len(findings) >= limit:
            break
    return findings[:limit]


@tool
@guard
def secret_scanner(path: str = ".", text: str = "", max_findings: int = 50, min_entropy: float = 4.2) -> str:
    """Scan workspace files (or pasted text) for leaked secrets: API keys, tokens, private keys, hardcoded passwords, high-entropy strings. Findings are redacted, never the full secret.

    Args:
        path: workspace file or folder to scan (default "."); ignored when `text` is given. Skips .git, node_modules, venvs, binaries and files over 1 MB.
        text: scan this text instead of files.
        max_findings: stop after this many findings (1-500, default 50).
        min_entropy: Shannon entropy (bits/char) threshold for the generic high-entropy rule (default 4.2; lower = more false positives).
    """
    limit = max(1, min(500, int(max_findings)))
    findings: List[dict] = []
    scanned = skipped = 0
    if text and text.strip():
        if len(text) > MAX_TEXT_CHARS:
            raise ToolError("text too long")
        findings = _scan_text(text, "<text>", limit, float(min_entropy))
        scanned = 1
    else:
        root = safe_path(path or ".", must_exist=True, allow_secret=True)
        files: List[Path] = []
        if root.is_file():
            files = [root]
        else:
            for dp, dns_, fns in os.walk(root):
                dns_[:] = sorted(d for d in dns_ if d not in _SKIP_DIRS)
                for fn in sorted(fns):
                    files.append(Path(dp) / fn)
                if len(files) > MAX_SCAN_FILES:
                    break
        ws = workspace_root()
        for f in files[:MAX_SCAN_FILES]:
            try:
                real = f.resolve()
                real.relative_to(ws)  # symlinks must not lead out of the workspace
                if f.is_symlink() or not real.is_file() or real.stat().st_size > MAX_SCAN_FILE_BYTES:
                    skipped += 1
                    continue
                raw = real.read_bytes()
            except (OSError, ValueError):
                skipped += 1
                continue
            if b"\x00" in raw[:4096]:
                skipped += 1
                continue
            scanned += 1
            rel = str(real.relative_to(ws))
            findings.extend(_scan_text(raw.decode("utf-8", errors="replace"), rel, limit - len(findings), float(min_entropy)))
            if len(findings) >= limit:
                break
    by_type = Counter(f["type"] for f in findings)
    return ok_json({
        "files_scanned": scanned, "files_skipped": skipped, "finding_count": len(findings),
        "truncated_at_limit": len(findings) >= limit, "by_type": dict(by_type), "findings": findings,
        "note": "pattern + entropy heuristics: expect some false positives and misses. Values are redacted; "
                "rotate any real secret found, removing it from git history is not enough.",
    })


# ---------------------------------------------------------------------------
# pii_redactor
# ---------------------------------------------------------------------------
def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_PHONE_RE = re.compile(r"(?<![\w+])(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<![\d-])(?<!\d[ -])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![\d-])(?![ -]\d)")
_PAN_RE = re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b")
_IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])|(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,7}:(?![\w:])")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b")


def redact_pii(text: str) -> tuple:
    counts: Counter = Counter()

    def card(m):
        if luhn_valid(m.group(0)):
            counts["card"] += 1
            return "[CARD]"
        return m.group(0)

    def simple(label):
        def f(m):
            counts[label] += 1
            return f"[{label.upper()}]"
        return f

    out = _CARD_RE.sub(card, text)
    out = _EMAIL_RE.sub(simple("email"), out)
    out = _PAN_RE.sub(simple("pan"), out)
    out = _PHONE_RE.sub(simple("phone"), out)
    out = _AADHAAR_RE.sub(simple("aadhaar"), out)
    out = _IPV4_RE.sub(simple("ip"), out)
    out = _IPV6_RE.sub(simple("ip"), out)
    return out, dict(counts)


@tool
@guard
def pii_redactor(text: str = "", path: str = "", output_path: str = "") -> str:
    """Redact personal data (emails, Indian phone numbers, Aadhaar-format numbers, PAN, Luhn-valid card numbers, IPv4/IPv6 addresses) with placeholders like [EMAIL]. Deterministic regex rules.

    Args:
        text: text to redact (use this or `path`).
        path: workspace file to redact instead of `text`.
        output_path: optional workspace file to write the redacted text to (the original is never modified).
    """
    if path and path.strip():
        src = read_text_file(safe_path(path, must_exist=True))
    else:
        src = text or ""
    if not src.strip():
        raise ToolError("no text given (provide `text` or a workspace file `path`)")
    if len(src) > MAX_TEXT_CHARS:
        raise ToolError("text too long")
    red, counts = redact_pii(src)
    result: Dict[str, object] = {"redacted_counts": counts, "total_redactions": sum(counts.values()),
                                 "note": "regex-based: Aadhaar is matched by format only (no Verhoeff check), phones by Indian "
                                         "mobile format; names, addresses and free-form IDs are NOT detected."}
    if output_path and output_path.strip():
        out = safe_output_path(output_path, "redacted.txt")
        if path and out == safe_path(path):
            raise ToolError("output_path must differ from the input file")
        atomic_write_text(out, red)
        result["output_file"] = str(out)
    else:
        result["redacted_text"] = red
    return ok_json(result)


# ---------------------------------------------------------------------------
# encryption_tool
# ---------------------------------------------------------------------------
_ENV_KEY = "AGENT_ENCRYPTION_KEY"
_PREFIX = "agt1"


def _fernet_key_or_none(val: str) -> Optional[bytes]:
    try:
        raw = base64.urlsafe_b64decode(val.encode())
    except (ValueError, TypeError):
        return None
    return val.encode() if len(raw) == 32 and len(val) == 44 else None


def _scrypt_key(passphrase: str, salt: bytes) -> bytes:
    dk = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2 ** 15, r=8, p=1, dklen=32, maxmem=128 * 1024 * 1024)
    return base64.urlsafe_b64encode(dk)


def _get_secret(password: str = "") -> str:
    val = password or os.environ.get(_ENV_KEY, "")
    if not val:
        raise ToolError(f"pass `password` or set the {_ENV_KEY} environment variable first (a Fernet key, or a passphrase of 8+ characters). "
                        "For a fresh key run: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    return val


def _encrypt(data: bytes, password: str = "") -> str:
    fernet_mod = require("cryptography.fernet", "cryptography")
    val = _get_secret(password)
    fk = _fernet_key_or_none(val)
    if fk:
        return f"{_PREFIX}:k:" + fernet_mod.Fernet(fk).encrypt(data).decode()
    if len(val) < 8:
        raise ToolError("the password/passphrase is shorter than 8 characters")
    salt = os.urandom(16)
    tok = fernet_mod.Fernet(_scrypt_key(val, salt)).encrypt(data).decode()
    return f"{_PREFIX}:p:{base64.urlsafe_b64encode(salt).decode()}:{tok}"


def _decrypt(blob: str, password: str = "") -> bytes:
    fernet_mod = require("cryptography.fernet", "cryptography")
    val = _get_secret(password)
    parts = blob.strip().split(":")
    try:
        if len(parts) == 3 and parts[0] == _PREFIX and parts[1] == "k":
            fk = _fernet_key_or_none(val)
            if not fk:
                raise ToolError("data was encrypted with a Fernet key but the supplied secret is a passphrase")
            return fernet_mod.Fernet(fk).decrypt(parts[2].encode())
        if len(parts) == 4 and parts[0] == _PREFIX and parts[1] == "p":
            salt = base64.urlsafe_b64decode(parts[2].encode())
            return fernet_mod.Fernet(_scrypt_key(val, salt)).decrypt(parts[3].encode())
    except fernet_mod.InvalidToken:
        raise ToolError("decryption failed: wrong key/passphrase or the data was modified")
    except ValueError:
        raise ToolError("encrypted data is malformed")
    raise ToolError("input is not data produced by this tool (expected an 'agt1:' envelope)")


@tool
@guard
def encryption_tool(action: str, text: str = "", path: str = "", output_path: str = "", overwrite: bool = False,
                    password: str = "") -> str:
    """Encrypt/decrypt text or workspace files with authenticated encryption (Fernet: AES-128-CBC + HMAC-SHA256). Secret = `password` argument, else env var AGENT_ENCRYPTION_KEY (Fernet key or passphrase, 8+ chars, stretched with scrypt + random salt). The password is a tool argument only because the tool has no other interactive channel: it travels through the LLM call, so prefer the env var for anything valuable. It is never logged, echoed or returned. Wrong password gives a clean error. Needs `cryptography`.

    Args:
        action: "encrypt_text", "decrypt_text", "encrypt_file" or "decrypt_file".
        text: text to encrypt, or the "agt1:..." envelope to decrypt (text actions).
        path: workspace file to encrypt/decrypt (file actions, up to 50 MB).
        output_path: where to write the result; default is path + ".enc" when encrypting, or path without ".enc" (else + ".dec") when decrypting.
        overwrite: allow replacing an existing output file (default False).
        password: password/passphrase (8+ chars) or Fernet key; empty = use AGENT_ENCRYPTION_KEY.
    """
    a = (action or "").strip().lower()
    if a not in ("encrypt_text", "decrypt_text", "encrypt_file", "decrypt_file"):
        raise ToolError("action must be encrypt_text, decrypt_text, encrypt_file or decrypt_file")
    if a == "encrypt_text":
        if not text:
            raise ToolError("`text` is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ToolError("text too long")
        return ok_json({"ciphertext": _encrypt(text.encode("utf-8"), password)})
    if a == "decrypt_text":
        if not text:
            raise ToolError("`text` is empty")
        return ok_json({"plaintext": _decrypt(text, password).decode("utf-8", errors="replace")})

    src = safe_path(path, must_exist=True)
    if not src.is_file():
        raise ToolError("path is not a file")
    if src.stat().st_size > MAX_CRYPT_BYTES:
        raise ToolError("file too large (limit 50 MB)")
    if output_path and output_path.strip():
        dst = safe_path(output_path)
    elif a == "encrypt_file":
        dst = src.with_name(src.name + ".enc")
    else:
        dst = src.with_suffix("") if src.suffix == ".enc" else src.with_name(src.name + ".dec")
    dst = safe_output_path(str(dst))
    if dst == src:
        raise ToolError("output must differ from the input file")
    if dst.exists() and not overwrite:
        raise ToolError(f"{dst.name} already exists; pass overwrite=true to replace it")
    data = src.read_bytes()
    if a == "encrypt_file":
        atomic_write_text(dst, _encrypt(data, password))
    else:
        atomic_write_bytes(dst, _decrypt(data.decode("ascii", errors="replace"), password))
    return ok_json({"action": a, "input": str(src), "output": str(dst), "bytes_written": dst.stat().st_size,
                    "note": "the original file is left untouched; keep your key/passphrase safe - without it the data is unrecoverable"})


SECURITY_PRIVACY_TOOLS = [password_generator, hash_tool, secret_scanner, pii_redactor, encryption_tool]
