#!/usr/bin/env python3
"""
patch_persist_state.py — Makes InternalState save/load from disk.
Run in the same directory as llm_modules.py
"""

with open('llm_modules.py') as f:
    code = f.read()

# Find and replace the InternalState __init__
OLD = '''    def __init__(self):
        self.state = {
            "frustration": 0.0,
            "satisfaction": 0.0,
            "curiosity": 0.0,
            "anxiety": 0.0,
            "excitement": 0.0,
            "warmth": 0.0,
        }
        self.history = []  # track state over time
        self._decay_rate = 0.85  # emotions fade gradually (like humans)'''

NEW = '''    def __init__(self, path="internal_state.json"):
        self.path = path
        self._decay_rate = 0.85
        try:
            with open(path, "r") as f:
                saved = json.load(f)
            self.state = saved.get("state", {})
            self.history = saved.get("history", [])
            # ensure all dimensions exist
            for dim in ["frustration","satisfaction","curiosity","anxiety","excitement","warmth"]:
                if dim not in self.state:
                    self.state[dim] = 0.0
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {
                "frustration": 0.0,
                "satisfaction": 0.0,
                "curiosity": 0.0,
                "anxiety": 0.0,
                "excitement": 0.0,
                "warmth": 0.0,
            }
            self.history = []'''

if OLD in code:
    code = code.replace(OLD, NEW)
    print("Fixed InternalState.__init__ to load from disk")
else:
    print("SKIP: InternalState.__init__ not found")

# Add save call to the snapshot method
OLD_SNAP = '''    def snapshot(self) -> dict:
        """Record current state for history tracking."""
        snap = {
            "state": dict(self.state),
            "dominant": self.get_dominant_emotion(),
            "timestamp": time.time(),
        }
        self.history.append(snap)
        # Keep history manageable
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return snap'''

NEW_SNAP = '''    def snapshot(self) -> dict:
        """Record current state for history tracking and save to disk."""
        snap = {
            "state": dict(self.state),
            "dominant": self.get_dominant_emotion(),
            "timestamp": time.time(),
        }
        self.history.append(snap)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self._save()
        return snap

    def _save(self):
        """Persist emotional state to disk."""
        data = {
            "state": self.state,
            "dominant": self.get_dominant_emotion(),
            "history": self.history[-50:],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)'''

if OLD_SNAP in code:
    code = code.replace(OLD_SNAP, NEW_SNAP)
    print("Added persistence to snapshot + _save method")
else:
    print("SKIP: snapshot method not found")

with open('llm_modules.py', 'w') as f:
    f.write(code)

print("Done — InternalState now persists across sessions")
