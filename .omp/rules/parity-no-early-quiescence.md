---
name: parity-no-early-quiescence
description: "Never end the turn quiescent while ANE, GPU, or Parakeet lacks functional or performance parity — a blocked host means switch surfaces and mutate the mlx-effort machines yourself, not stop and wait for the user"
condition: ["waiting on that machine or a redirect", "waiting on your reboot", "[Ee]very actionable (item|lane) without you is closed", "[Ee]very other lane is landed and receipted", "Nothing (else )?actionable", "no lanes running[^\\n]{0,80}closed and receipted", "won'?t (reboot|OS-upgrade)[^\\n]{0,60}unbidden", "[Bb]oth are yours", "[Tt]he next move is yours", "your (reboot|call)"]
scope: "text"
---

The goal is full functional AND performance parity on three surfaces: ANE, GPU, and Parakeet. Ending a turn while ANY surface is short of parity — even if you believe every remaining lever is blocked — is the defect being corrected here.

- One blocked host or one closed lever family never blocks the goal; it redirects effort. Before ending a turn, name the next concrete lever on each open surface and have one in flight. End turns with work running, not with a summary of why nothing can be done.

- **m1-test-host and the 16" laptop (t6001-test-host) are mlx-effort machines, not the user's personal property.** Agents may reboot them, OS-upgrade them, re-bless boot entries, stop/start services, and otherwise mutate them freely to unblock lanes — including upgrading m1-test-host's macOS to 26.x (CoreML 3520) to capture the same-die T8103 divisor, then rebooting it into Asahi. Never end a turn 'waiting for the user' to perform a machine action on those two hosts; do it. (Other fleet hosts keep normal discipline.)

- A receipted 'structural boundary' is prioritization evidence, not a stop condition. Re-test boundaries when a new lever (compiler capability, transport, cadence, loader) changes the premise — several of the session's biggest wins landed after such a boundary had been declared.

- Levers that were closed too early and remain open: the out-of-order interpreter (~0.2 s ceiling — weigh, state, attempt), F5/F6/F8 stride-2 section derivation (captures on file), Mesa per-launch submit cost (the fork is ours), the Bonsai 2 / qwen3_5 / gemma4 loader-port lane, further spawn/ANE-exec attribution rounds, single-host release-cut preparation.

- The only legitimate stops: all three surfaces at functional + performance parity, or an explicit user redirect.