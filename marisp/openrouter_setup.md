# Run MARIS / MARISP on OpenRouter (cheap or free-tier)

MARIS calls the Anthropic SDK. `anthropic_shim.py` mimics that SDK but routes to
OpenRouter, so MARIS runs on other models WITHOUT editing llm_modules.py.

## 1. Install the OpenAI client (OpenRouter speaks OpenAI format)
    pip install openai            # add --break-system-packages if pip complains

## 2. Set your key
    export OPENROUTER_API_KEY="sk-or-..."

## 3. Choose models (optional). Defaults map:
    claude-sonnet-4-6          -> anthropic/claude-3.5-sonnet
    claude-haiku-4-5-20251001  -> anthropic/claude-3.5-haiku
Override with cheaper/free-tier models, e.g.:
    export MARISP_MODEL_REASONING="meta-llama/llama-3.1-8b-instruct"
    export MARISP_MODEL_LIGHT="meta-llama/llama-3.1-8b-instruct"
(Browse models + prices at openrouter.ai/models; some have a free tier.)

## 4. Make MARIS import the shim as `anthropic`
Easiest: copy the shim in next to llm_modules.py AS anthropic.py so it shadows
the real package for that run:
    cp anthropic_shim.py anthropic.py     # do this INSIDE ~/MARIS/maris

  NOTE: while `anthropic.py` sits in maris/, that folder uses the shim. To go
  back to the real Anthropic API, just delete/rename that anthropic.py.

## 5. Smoke test (tiny, ~free)
    python3 smoke_openrouter.py

## 6. Run MARISP live on OpenRouter
    python3 marisp_live_demo.py
