"""Tiny OpenRouter smoke test via the shim. Confirms key + routing work cheaply."""
import sys
try:
    import anthropic_shim as shim
except ImportError:
    print("Put anthropic_shim.py in this folder first."); sys.exit(1)

client = shim.Anthropic()
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",   # gets mapped to your MARISP_MODEL_LIGHT
    max_tokens=20,
    messages=[{"role": "user", "content": "reply with exactly: ok"}],
)
print("OpenRouter replied:", resp.content[0].text.strip())
print("\nIf you see a reply above, the shim + your key work. MARISP can now run.")
