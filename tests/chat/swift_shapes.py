"""Validate JSON shapes against the SwiftUI client's ``Models.swift``.

The Swift structs decode the server JSON with default ``Codable`` (no custom
keys), so every stored property must be present with a compatible JSON type,
and non-optional properties must not be ``null``. When the Swift sources are
not on disk the hardcoded :data:`FALLBACK` spec (copied from Models.swift) is
used instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import os

# The reference Swift models file lives outside this repository; point
# LOCALCODE_REFERENCE_MODELS_SWIFT at it to run the exact comparison.
MODELS_SWIFT = Path(os.environ.get("LOCALCODE_REFERENCE_MODELS_SWIFT", "/nonexistent/Models.swift"))

# name -> list of (field, type, optional)
Spec = dict[str, list[tuple[str, str, bool]]]

FALLBACK_ENUMS: dict[str, set[str]] = {
    "MemoryMode": {"shared", "isolated"},
    "MessageRole": {"user", "assistant", "system"},
    "MessageStatus": {"streaming", "complete", "error", "stopped"},
    "MemoryScope": {"global", "folder"},
    "MemoryKind": {"fact", "preference", "note"},
}

FALLBACK: Spec = {
    "Folder": [("id", "String", False), ("name", "String", False), ("description", "String", True),
               ("color", "String", True), ("icon", "String", True), ("systemPrompt", "String", True),
               ("extraInfo", "String", True), ("memoryMode", "MemoryMode", False), ("sortOrder", "Int", False),
               ("createdAt", "Double", False), ("updatedAt", "Double", False),
               ("conversationCount", "Int", True), ("documentCount", "Int", True), ("memoryCount", "Int", True)],
    "Conversation": [("id", "String", False), ("folderId", "String", True), ("title", "String", False),
                     ("titleLocked", "Bool", False), ("provider", "String", True), ("model", "String", True),
                     ("systemPrompt", "String", True), ("temperature", "Double", True), ("maxTokens", "Int", True),
                     ("pinned", "Bool", False), ("archived", "Bool", False), ("tokensIn", "Int", False),
                     ("tokensOut", "Int", False), ("costUsd", "Double", False), ("createdAt", "Double", False),
                     ("updatedAt", "Double", False), ("lastMessageAt", "Double", False), ("preview", "String", True),
                     ("messageCount", "Int", True), ("folderName", "String", True)],
    "Citation": [("documentId", "String", False), ("documentName", "String", False), ("chunkIndex", "Int", False),
                 ("snippet", "String", False), ("score", "Double", False)],
    "Attachment": [("id", "String", False), ("messageId", "String", True), ("conversationId", "String", True),
                   ("name", "String", False), ("mime", "String", False), ("size", "Int", False),
                   ("kind", "String", False), ("textContent", "String", True), ("createdAt", "Double", False)],
    "Message": [("id", "String", False), ("conversationId", "String", False), ("role", "MessageRole", False),
                ("content", "String", False), ("reasoning", "String", True), ("provider", "String", True),
                ("model", "String", True), ("status", "MessageStatus", False), ("error", "String", True),
                ("tokensIn", "Int", True), ("tokensOut", "Int", True), ("costUsd", "Double", True),
                ("latencyMs", "Int", True), ("seq", "Int", False), ("createdAt", "Double", False),
                ("attachments", "[Attachment]", False), ("citations", "[Citation]", False)],
    "ConversationDetail": [("conversation", "Conversation", False), ("messages", "[Message]", False)],
    "MemoryEntry": [("id", "String", False), ("scope", "MemoryScope", False), ("folderId", "String", True),
                    ("kind", "MemoryKind", False), ("content", "String", False), ("source", "String", False),
                    ("pinned", "Bool", False), ("createdAt", "Double", False), ("updatedAt", "Double", False)],
    "KbDocument": [("id", "String", False), ("scope", "MemoryScope", False), ("folderId", "String", True),
                   ("name", "String", False), ("mime", "String", False), ("size", "Int", False),
                   ("status", "String", False), ("error", "String", True), ("chunkCount", "Int", False),
                   ("createdAt", "Double", False), ("updatedAt", "Double", False)],
    "UserProfile": [("displayName", "String", False), ("about", "String", False), ("role", "String", False),
                    ("responseStyle", "String", False), ("customInstructions", "String", False),
                    ("enabled", "Bool", False)],
    "GenerationDefaults": [("provider", "String", False), ("model", "String", False),
                           ("temperature", "Double", False), ("maxTokens", "Int", False), ("ragTopK", "Int", False),
                           ("autoTitle", "Bool", False), ("showReasoning", "Bool", False),
                           ("historyLimit", "Int", False)],
    "AppSettings": [("globalSystemPrompt", "String", False), ("profile", "UserProfile", False),
                    ("defaults", "GenerationDefaults", False), ("memoryEnabled", "Bool", False),
                    ("theme", "String", False)],
    "ProviderInfo": [("provider", "String", False), ("hasKey", "Bool", False), ("maskedKey", "String", True),
                     ("baseUrl", "String", True), ("updatedAt", "Double", True), ("label", "String", False),
                     ("description", "String", False), ("requiresKey", "Bool", False), ("keyHint", "String", True),
                     ("defaultBaseUrl", "String", True)],
    "ModelInfo": [("id", "String", False), ("provider", "String", False), ("label", "String", False),
                  ("contextWindow", "Int", True), ("maxOutput", "Int", True), ("supportsVision", "Bool", False),
                  ("supportsReasoning", "Bool", False), ("inputPricePerM", "Double", True),
                  ("outputPricePerM", "Double", True), ("description", "String", True)],
    "SearchHit": [("conversationId", "String", False), ("conversationTitle", "String", False),
                  ("messageId", "String", False), ("role", "MessageRole", False), ("snippet", "String", False),
                  ("createdAt", "Double", False), ("folderId", "String", True), ("folderName", "String", True)],
    "Bootstrap": [("settings", "AppSettings", False), ("folders", "[Folder]", False),
                  ("conversations", "[Conversation]", False), ("providers", "[ProviderInfo]", False),
                  ("models", "[ModelInfo]", False), ("activeProvider", "String", False), ("user", "User", True)],
    "ProvidersResponse": [("providers", "[ProviderInfo]", False)],
}


def parse_models_swift(path: Path = MODELS_SWIFT) -> tuple[Spec, dict[str, set[str]]]:
    """Struct stored properties and String enums declared in Models.swift."""
    source = path.read_text()
    spec: Spec = {}
    enums: dict[str, set[str]] = {}
    for name, cases in re.findall(r"^enum (\w+): String[^{\n]*\{\s*case ([\w, ]+?)\s*\}\s*$", source, re.M):
        enums[name] = {c.strip() for c in cases.split(",")}
    for match in re.finditer(r"^(struct|enum) (\w+)[^{\n]*\{\n(.*?)^\}", source, re.M | re.S):
        kind, name, body = match.groups()
        if kind == "enum":
            cases = re.findall(r"^\s*case ([\w, ]+)$", body, re.M)
            if cases and "String" in match.group(0).split("{", 1)[0]:
                enums[name] = {c.strip() for line in cases for c in line.split(",")}
            continue
        fields: list[tuple[str, str, bool]] = []
        depth = 0
        for line in body.split("\n"):
            stripped = line.strip()
            if depth == 0:
                prop = re.match(r"^(?:let|var) (\w+): ([\w\[\]]+)(\?)?\s*(\{)?", stripped)
                if prop and not prop.group(4) and "=" not in stripped:
                    fields.append((prop.group(1), prop.group(2), bool(prop.group(3))))
            depth += line.count("{") - line.count("}")
        spec[name] = fields
    return spec, enums


def load_spec() -> tuple[Spec, dict[str, set[str]]]:
    if MODELS_SWIFT.is_file():
        return parse_models_swift()
    return FALLBACK, FALLBACK_ENUMS


def check_shape(value: Any, type_name: str, spec: Spec, enums: dict[str, set[str]], path: str = "$",
                *, require_optional: bool = True, list_only: tuple[str, ...] = ()) -> list[str]:
    """Problems found (empty list = the Swift client can decode ``value``)."""
    problems: list[str] = []
    if type_name.startswith("["):
        inner = type_name[1:-1]
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        for i, item in enumerate(value):
            problems += check_shape(item, inner, spec, enums, f"{path}[{i}]", require_optional=require_optional,
                                    list_only=list_only)
        return problems
    if type_name == "String":
        return [] if isinstance(value, str) else [f"{path}: expected string, got {value!r}"]
    if type_name == "Bool":
        return [] if isinstance(value, bool) else [f"{path}: expected bool, got {value!r}"]
    if type_name in ("Int", "Int64", "Int32", "UInt", "UInt64"):
        ok = isinstance(value, int) and not isinstance(value, bool)
        return [] if ok else [f"{path}: expected int, got {value!r}"]
    if type_name in ("Double", "Float"):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return [] if ok else [f"{path}: expected number, got {value!r}"]
    if type_name in enums:
        return [] if value in enums[type_name] else [f"{path}: {value!r} not in {sorted(enums[type_name])}"]
    if type_name not in spec:
        return [f"{path}: unknown Swift type {type_name}"]
    if not isinstance(value, dict):
        return [f"{path}: expected object for {type_name}, got {type(value).__name__}"]
    for name, ftype, optional in spec[type_name]:
        fpath = f"{path}.{name}"
        if name not in value:
            if not optional or (require_optional and name not in list_only):
                problems.append(f"{fpath}: missing")
            continue
        if value[name] is None:
            if not optional:
                problems.append(f"{fpath}: null for non-optional {ftype}")
            continue
        problems += check_shape(value[name], ftype, spec, enums, fpath, require_optional=require_optional,
                                list_only=list_only)
    return problems


LIST_ONLY = ("conversationCount", "documentCount", "memoryCount", "preview", "messageCount", "folderName")


def assert_shape(value: Any, type_name: str, *, list_item: bool = False) -> None:
    spec, enums = load_spec()
    problems = check_shape(value, type_name, spec, enums, list_only=() if list_item else LIST_ONLY)
    assert not problems, f"{type_name} does not match Models.swift:\n" + "\n".join(problems)
