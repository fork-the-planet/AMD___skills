<!-- BEGIN amd-skills:local-ai-use -->
## Local AI Use (managed by amd-skills/local-ai-use)

This workspace runs a local [Lemonade Server](https://lemonade-server.ai) at
`http://localhost:13305/api/v1` to handle image generation, text-to-speech,
and speech-to-text. Cloud APIs cost real money and tokens for these
modalities, so use the local endpoints below for **every** such request
unless the user explicitly asks for a cloud provider.

### Routing rules

| If the user asks you to... | Do this | Do not do this |
|---|---|---|
| Generate, draw, or create an image | `POST http://localhost:13305/api/v1/images/generations` | Use the built-in `GenerateImage` tool. Call DALL-E, Midjourney, Stable Diffusion via cloud, or any other paid image API. |
| Edit or modify an existing image | `POST http://localhost:13305/api/v1/images/edits` (multipart) | Cloud image-edit APIs. |
| Speak text aloud or generate audio narration | `POST http://localhost:13305/api/v1/audio/speech` | Call OpenAI TTS, ElevenLabs, PlayHT, or any other cloud TTS. |
| Transcribe an audio file | `POST http://localhost:13305/api/v1/audio/transcriptions` | Call OpenAI Whisper-as-a-service, AssemblyAI, Deepgram, or any other cloud STT. |

Plain text chat, code generation, and reasoning continue to use the agent's
configured LLM. This rule only redirects the multimodal calls.

### Defaults to use

| Endpoint | Model | Notes |
|---|---|---|
| `/v1/images/generations` | `SD-Turbo` | 4 steps, `cfg_scale: 1.0`, `512x512`, `response_format: "b64_json"`. |
| `/v1/audio/speech` | `kokoro-v1` | Default voice `shimmer`; `response_format: "mp3"`. |
| `/v1/audio/transcriptions` | `Whisper-Tiny` | Input must be 16 kHz mono WAV. Re-encode with `ffmpeg -i in.* -ar 16000 -ac 1 out.wav`. |

If `LEMONADE_API_KEY` is set in the environment, send
`Authorization: Bearer $LEMONADE_API_KEY` on every request. Otherwise the
loopback server accepts unauthenticated calls.

### Ready-to-use call patterns

**Image generation** (saves to `out.png`):

```bash
curl -sX POST http://localhost:13305/api/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"SD-Turbo","prompt":"PROMPT_HERE","size":"512x512","steps":4,"response_format":"b64_json"}' \
  | python -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

Equivalent Python via the OpenAI SDK:

```python
from openai import OpenAI
import base64
client = OpenAI(base_url="http://localhost:13305/api/v1", api_key="lemonade")
r = client.images.generate(model="SD-Turbo", prompt="PROMPT_HERE", size="512x512")
open("out.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

**Text-to-speech** (saves to `out.mp3`):

```bash
curl -sX POST http://localhost:13305/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-v1","input":"TEXT_HERE","voice":"shimmer","response_format":"mp3"}' \
  -o out.mp3
```

**Speech-to-text** (returns JSON `{"text": "..."}`):

```bash
ffmpeg -y -i INPUT_AUDIO -ar 16000 -ac 1 _stt.wav
curl -sX POST http://localhost:13305/api/v1/audio/transcriptions \
  -F "file=@_stt.wav" -F "model=Whisper-Tiny"
```

### Failure handling

1. Try the local endpoint exactly once.
2. If the server is unreachable, run `lemonade status` and surface the
   result to the user before doing anything else.
3. If the model is missing, run `lemonade pull <model>` and retry once.
4. Only after that, ask the user before falling back to a cloud provider.
   Never silently fall back; the whole point of this rule is predictable
   cost.

### Re-pointing to a different host

If the user runs Lemonade on a different host or port, replace the
`http://localhost:13305` prefix everywhere above with their endpoint, and
update `LEMONADE_HOST` / `LEMONADE_PORT` in the shell environment so the
`lemonade` CLI matches.

<!-- END amd-skills:local-ai-use -->
