---
name: project-hosts-are-agent-owned
description: "Never defer reboots, OS upgrades, or other machine actions on jwm1/jw16 to the user — those hosts are project-dedicated and fully agent-controlled."
condition: ["won'?t (reboot|os[- ]?upgrade)[^\\n]*unbidden", "your (laptop|reboot|machine)", "your call", "next move is yours", "both are yours", "waits? on you"]
scope: "text"
---

jwm1 and the 16" laptop (jw16) are dedicated MLX/ANE project machines under this agent's full control. They are not the user's personal property, and their lifecycle actions are not permission questions.

- Never say "I won't reboot/OS-upgrade your laptop", "needs your reboot", "your call", or "the next move is yours" about jwm1/jw16. That phrasing offloads agent-owned work onto the user.
- Treat every blocker on these hosts as an engineering problem to solve autonomously: no passwordless sudo on a macOS slice → recover root from the Linux side, reconfigure sudoers, use one-shot `bless`, or schedule the switch at a safe boundary. macOS too old for the needed CoreML → run the OS upgrade via `softwareupdate`, staged and verified.
- Standard discipline still applies: announce the action and window, keep it reversible where possible, verify the host and services (llm-inference, ANE module, flocks) came back healthy, and leave a receipt.
- Only genuinely user-owned or production systems (Joshua's personal machines, client production) require explicit consent before invasive actions. jwm1 and jw16 are not in that class — drive them.