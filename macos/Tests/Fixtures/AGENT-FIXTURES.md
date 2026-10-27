# Fixtury agenta kodującego (`agent-*`)

Nagrane z **działającego** serwera (`.venv/bin/localcode serve --port 0 --token test`)
i **prawdziwego** modelu `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M`
(GGUF, n_ctx 8192) w katalogu tymczasowym. Każda linia `.jsonl` to jedna ramka
WebSocketu `GET /api/agent/sessions/{id}/ws`, dokładnie w kolejności nadania.

| plik | co zawiera |
|---|---|
| `agent-turn-yolo.jsonl` | pełna tura w trybie `yolo`: `list_dir` + `write_file` z diffem, strumień `reasoning_delta`/`text_delta`, `turn_end` ze statystykami |
| `agent-turn-permission-write.jsonl` | tryb `default`: `permission_request` (podgląd = diff) → `permission_resolved: allow` → `tool_result` |
| `agent-turn-bash-live-output.jsonl` | `bash` z listy tylko-do-odczytu: ramki `status` z `phase: "tool"` niosące wyjście polecenia linia po linii |
| `agent-turn-deny-interrupt.jsonl` | trzy odmowy (`deny`), powtórzone wywołanie, `todos`, wiadomość w kolejce (`ack: queued`) i `turn_end` z `reason: "interrupted"` |
| `agent-turn-compaction.jsonl` | prawdziwa kompaktacja na wznowionej, długiej sesji: `mask_tool_results` 15581 → 7829 i `summarize` 7829 → 4020 |
| `agent-session-snapshot.json` | odpowiedź `GET /api/agent/sessions/{id}` — `{session, items, todos}` |
| `agent-events-synthetic.jsonl` | ramki **napisane ręcznie** wg `localcode/events.py` dla przypadków, których model 0.6B nie wywołał podczas nagrywania (status ładowania modelu, plan z trzema stanami, prośba o zgodę na polecenie, błąd, `max_steps`) |

W plikach nagranych tury `text_delta`/`reasoning_delta` zostały przycięte do
pierwszych ~120–150 ramek — reszta to setki jednotokenowych ramek tego samego
typu, które niczego nie dowodzą, a ważyły 10× tyle.

Fixtury są odtwarzane przez `AgentEventDecoder.decode` i `TranscriptBuilder`
(`macos/Sources/CodeFeature/Models/`). Obie klasy zależą wyłącznie od
`Foundation`, więc można je skompilować bez SwiftUI, np.:

```
xcrun swiftc -swift-version 5 \
  macos/Sources/CodeFeature/Models/{AgentModels,AgentEvent,Transcript}.swift \
  replay.swift -o replay && ./replay
```

Stan na dzień nagrania: wszystkie 1365 ramek dekodują się bez `unknown`.
