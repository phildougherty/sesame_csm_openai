# CSM-1B TTS API

An OpenAI-compatible Text-to-Speech API that harnesses the power of Sesame's Conversational Speech Model (CSM-1B). This API allows you to generate high-quality speech from text using a variety of consistent voices, compatible with systems like OpenWebUI, ChatBot UI, and any platform that supports the OpenAI TTS API format.

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI's TTS API
- **Multiple Voices**: Six distinct voices (alloy, echo, fable, onyx, nova, shimmer)
- **Voice Consistency**: Maintains consistent voice characteristics across multiple requests
- **Voice Cloning**: Clone your own voice from audio samples
- **Conversational Context**: Supports conversational context for improved naturalness
- **Multiple Audio Formats**: Supports MP3, OPUS, AAC, FLAC, and WAV
- **Speed Control**: Adjustable speech speed
- **CUDA Acceleration**: GPU support for faster generation
- **Web UI**: Simple interface for voice cloning and speech generation

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Hugging Face account with access to `sesame/csm-1b` model

### Installation

1. Clone this repository:
```bash
git clone https://github.com/phildougherty/sesame_csm_openai
cd sesame_csm_openai
```

2. Create a `.env` file in the /app folder with your Hugging Face token:
```
HF_TOKEN=your_hugging_face_token_here
```

3. Build and start the container:
```bash
docker compose up -d --build
```

The server will start on port 8000. First startup may take some time as it downloads the model files.

## Hugging Face Configuration (ONLY NEEDED TO ACCEPT TERMS/DOWNLOAD MODEL)

This API requires access to the `sesame/csm-1b` model on Hugging Face:

1. Create a Hugging Face account if you don't have one: [https://huggingface.co/join](https://huggingface.co/join)
2. Accept the model license at [https://huggingface.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
3. Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Use this token in your `.env` file or pass it directly when building the container:

```bash
HF_TOKEN=your_token docker compose up -d --build
```

### Required Models

The API uses the following models which are downloaded automatically:

- **CSM-1B**: The main speech generation model from Sesame
- **Mimi**: Audio codec for high-quality audio generation
- **Llama Tokenizer**: Uses the unsloth/Llama-3.2-1B tokenizer for text processing

## Multi-GPU Support

The CSM-1B model can be distributed across multiple GPUs to handle larger models or improve performance. To enable multi-GPU support, set the `CSM_DEVICE_MAP` environment variable:

```bash
# Automatic device mapping (recommended)
CSM_DEVICE_MAP=auto docker compose up -d

# Balanced distribution of layers across GPUs
CSM_DEVICE_MAP=balanced docker compose up -d

# Sequential distribution (backbone on first GPUs, decoder on remaining)
CSM_DEVICE_MAP=sequential docker compose up -d
```

## Real-Time Streaming TTS

CSM-1B TTS API supports real-time streaming speech synthesis, allowing audio to be delivered and played as it's being generated, rather than waiting for the entire generation to complete. This is particularly useful for longer text or responsive user interfaces.

### Streaming API Endpoints

The API provides two endpoints for real-time streaming:

1. **OpenAI-Compatible Streaming**: `/v1/audio/speech/streaming`
   - Follows the same format as the OpenAI streaming TTS API
   - Drop-in replacement for OpenAI's streaming endpoint

2. **Standard Streaming**: `/v1/audio/speech/stream`
   - CSM-specific streaming implementation
   - Offers the same functionality but with a different URL path

### How Streaming Works

1. **Chunked Generation**: Text is split into small segments
2. **Incremental Processing**: Each segment is processed in sequence
3. **Real-Time Delivery**: Audio is delivered as it's generated
4. **Client-Side Playback**: Audio is played as it arrives

### Streaming API Example

```bash
curl -X POST http://localhost:8000/v1/audio/speech/streaming \
  -H "Content-Type: application/json" \
  -d '{
    "model": "csm-1b",
    "input": "This is a streaming text-to-speech example that starts playing before the entire text is processed.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output - | ffplay -autoexit -
```

For web applications, you can use the Web Audio API to play the streaming audio:

```javascript
async function streamSpeech(text, voice) {
  const response = await fetch('/v1/audio/speech/streaming', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'csm-1b',
      input: text,
      voice: voice,
      response_format: 'mp3'
    })
  });

  if (!response.ok) {
    throw new Error('Streaming request failed');
  }

  // For browsers that support it
  const audio = new Audio();
  audio.src = URL.createObjectURL(await response.blob());
  audio.play();
}
```

### Streaming Demo

The API includes a streaming demo page at `http://your-server-ip:8000/streaming-demo` where you can test real-time streaming speech generation with various voices and settings.

### Supported Streaming Formats

The streaming API supports the following formats:

- MP3 (audio/mpeg)
- Opus (audio/opus)
- AAC (audio/aac)
- FLAC (audio/flac)
- WAV (audio/wav, recommended for lowest latency)

### Benefits of Streaming

- **Lower Perceived Latency**: Audio begins playing immediately
- **Better User Experience**: No waiting for long texts
- **Memory Efficiency**: Processing text in chunks uses less memory
- **Responsive UI**: Applications remain responsive during generation

## Voice Cloning Guide

The CSM-1B TTS API comes with powerful voice cloning capabilities that allow you to create custom voices from audio samples. Here's how to use this feature:

### Method 1: Using the Web Interface

1. Access the voice cloning UI by navigating to `http://your-server-ip:8000/voice-cloning` in your browser.

2. **Clone a Voice**:
   - Go to the "Clone Voice" tab
   - Enter a name for your voice
   - Upload an audio sample (2-3 minutes of clear speech works best)
   - Optionally provide a transcript of the audio for better results
   - Click "Clone Voice"

3. **View Your Voices**:
   - Navigate to the "My Voices" tab to see all your cloned voices
   - You can preview or delete voices from this tab

4. **Generate Speech**:
   - Go to the "Generate Speech" tab
   - Select one of your cloned voices
   - Enter the text you want to synthesize
   - Adjust the temperature slider if needed (lower for more consistent results)
   - Click "Generate Speech" and listen to the result

### Method 2: Using the API

1. **Clone a Voice**:
```bash
curl -X POST http://localhost:8000/v1/voice-cloning/clone \
  -F "name=My Voice" \
  -F "audio_file=@path/to/your/voice_sample.mp3" \
  -F "transcript=Optional transcript of the audio sample" \
  -F "description=A description of this voice"
```

2. **List Available Cloned Voices**:
```bash
curl -X GET http://localhost:8000/v1/voice-cloning/voices
```

3. **Generate Speech with a Cloned Voice**:
```bash
curl -X POST http://localhost:8000/v1/voice-cloning/generate \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "1234567890_my_voice",
    "text": "This is my cloned voice speaking.",
    "temperature": 0.7
  }' \
  --output cloned_speech.mp3
```

4. **Generate a Voice Preview**:
```bash
curl -X POST http://localhost:8000/v1/voice-cloning/voices/1234567890_my_voice/preview \
  --output voice_preview.mp3
```

5. **Delete a Cloned Voice**:
```bash
curl -X DELETE http://localhost:8000/v1/voice-cloning/voices/1234567890_my_voice
```

### Voice Cloning Best Practices

For the best voice cloning results:

1. **Use High-Quality Audio**: Record in a quiet environment with minimal background noise and echo.

2. **Provide Sufficient Length**: 2-3 minutes of speech provides better results than shorter samples.

3. **Clear, Natural Speech**: Speak naturally at a moderate pace with clear pronunciation.

4. **Include Various Intonations**: Sample should contain different sentence types (statements, questions) for better expressiveness.

5. **Add a Transcript**: While optional, providing an accurate transcript of your recording helps the model better capture your voice characteristics.

6. **Adjust Temperature**: For more consistent results, use lower temperature values (0.6-0.7). For more expressiveness, use higher values (0.7-0.9).

7. **Try Multiple Samples**: If you're not satisfied with the results, try recording a different sample or adjusting the speaking style.

### Using Cloned Voices with the Standard TTS Endpoint

Cloned voices are automatically available through the standard OpenAI-compatible endpoint. Simply use the voice ID or name as the `voice` parameter:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "csm-1b",
    "input": "This is my cloned voice speaking through the standard endpoint.",
    "voice": "1234567890_my_voice",
    "response_format": "mp3"
  }' \
  --output cloned_speech.mp3
```

## YouTube Voice Cloning 

The CSM-1B TTS API now includes the ability to clone voices directly from YouTube videos. This feature allows you to extract voice characteristics from any YouTube content and create custom TTS voices without needing to download or prepare audio samples yourself.

## How to Clone a Voice from YouTube

### API Endpoint

```
POST /v1/audio/speech/voice-cloning/youtube
```

Parameters:
- `youtube_url`: URL of the YouTube video
- `voice_name`: Name for the cloned voice
- `start_time` (optional): Start time in seconds (default: 0)
- `duration` (optional): Duration to extract in seconds (default: 180)
- `description` (optional): Description of the voice

Example request:
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "voice_name": "rick_astley",
  "start_time": 30,
  "duration": 60,
  "description": "Never gonna give you up"
}
```

Response:
```json
{
  "voice_id": "1710805983_rick_astley",
  "name": "rick_astley",
  "description": "Never gonna give you up",
  "created_at": "2025-03-18T22:53:03Z",
  "audio_duration": 60.0,
  "sample_count": 1440000
}
```

## How It Works

1. The system downloads the audio from the specified YouTube video
2. It extracts the specified segment (start time and duration)
3. Whisper ASR generates a transcript of the audio for better voice matching
4. The audio is processed to remove noise and silence
5. The voice is cloned and made available for TTS generation

## Best Practices for YouTube Voice Cloning

For optimal results:

1. **Choose Clear Speech Segments**
   - Select portions of the video with clear, uninterrupted speech
   - Avoid segments with background music, sound effects, or multiple speakers

2. **Optimal Duration**
   - 30-60 seconds of clean speech typically provides the best results
   - Longer isn't always better - quality matters more than quantity

3. **Specify Time Ranges Precisely**
   - Use `start_time` and `duration` to target the exact speech segment
   - Preview the segment in YouTube before cloning to ensure it's suitable

4. **Consider Audio Quality**
   - Higher quality videos generally produce better voice clones
   - Interviews, vlogs, and speeches often work better than highly produced content

## Limitations

- YouTube videos with heavy background music may result in lower quality voice clones
- Very noisy or low-quality audio sources will produce less accurate voice clones
- The system works best with natural speech rather than singing or exaggerated voices
- Copyright restrictions apply - only clone voices you have permission to use

## Example Use Cases

- Create a voice clone of a public figure for educational content
- Clone your own YouTube voice for consistent TTS across your applications
- Create voice clones from historical speeches or interviews (public domain)
- Develop custom voices for creative projects with proper permissions

## Ethical Considerations

Please use YouTube voice cloning responsibly:
- Only clone voices from content you have permission to use
- Respect copyright and intellectual property rights
- Clearly disclose when using AI-generated or cloned voices
- Do not use cloned voices for impersonation, deception, or harmful content

## How the Voices Work

Unlike traditional TTS systems with pre-trained voice models, CSM-1B works differently:

- The base CSM-1B model is capable of producing a wide variety of voices but doesn't have fixed voice identities
- This API creates consistent voices by using acoustic "seed" samples for each named voice
- When you specify a voice (e.g., "alloy"), the API uses a consistent acoustic seed and speaker ID
- The most recent generated audio becomes the new reference for that voice, maintaining voice consistency
- Each voice has unique tonal qualities:
  - **alloy**: Balanced mid-tones with natural inflection
  - **echo**: Resonant with slight reverberance
  - **fable**: Brighter with higher pitch
  - **onyx**: Deep and resonant
  - **nova**: Warm and smooth
  - **shimmer**: Light and airy with higher frequencies

The voice system can be extended with your own voice samples by using the voice cloning feature.

## API Usage

### Basic Usage

Generate speech with a POST request to `/v1/audio/speech`:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "csm-1b",
    "input": "Hello, this is a test of the CSM text to speech system.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Available Endpoints

#### Standard TTS Endpoints
- `GET /v1/audio/models` - List available models
- `GET /v1/audio/voices` - List available voices (including cloned voices)
- `GET /v1/audio/speech/response-formats` - List available response formats
- `POST /v1/audio/speech` - Generate speech from text
- `POST /api/v1/audio/conversation` - Advanced endpoint for conversational speech

#### Voice Cloning Endpoints
- `POST /v1/voice-cloning/clone` - Clone a new voice from an audio sample
- `GET /v1/voice-cloning/voices` - List all cloned voices
- `POST /v1/voice-cloning/generate` - Generate speech with a cloned voice
- `POST /v1/voice-cloning/voices/{voice_id}/preview` - Generate a preview of a cloned voice
- `DELETE /v1/voice-cloning/voices/{voice_id}` - Delete a cloned voice

### Request Parameters

#### Standard TTS
| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `model` | Model ID to use | string | "csm-1b" |
| `input` | The text to convert to speech | string | Required |
| `voice` | The voice to use (standard or cloned voice ID) | string | "alloy" |
| `response_format` | Audio format | string | "mp3" |
| `speed` | Speech speed multiplier | float | 1.0 |
| `temperature` | Sampling temperature | float | 0.8 |
| `max_audio_length_ms` | Maximum audio length in ms | integer | 90000 |

#### Voice Cloning
| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `name` | Name for the cloned voice | string | Required |
| `audio_file` | Audio sample file | file | Required |
| `transcript` | Transcript of the audio | string | Optional |
| `description` | Description of the voice | string | Optional |

### Available Voices

- `alloy` - Balanced and natural
- `echo` - Resonant
- `fable` - Bright and higher-pitched
- `onyx` - Deep and resonant
- `nova` - Warm and smooth
- `shimmer` - Light and airy
- `[cloned voice ID]` - Any voice you've cloned using the voice cloning feature

### Response Formats

- `mp3` - MP3 audio format
- `opus` - Opus audio format
- `aac` - AAC audio format
- `flac` - FLAC audio format
- `wav` - WAV audio format

## Integration with OpenWebUI

OpenWebUI is a popular open-source UI for AI models that supports custom TTS endpoints. Here's how to integrate the CSM-1B TTS API:

1. Access your OpenWebUI settings
2. Navigate to the TTS settings section
3. Select "Custom TTS Endpoint"
4. Enter your CSM-1B TTS API URL: `http://your-server-ip:8000/v1/audio/speech`
5. Use the API Key field to add any authentication if you've configured it (not required by default)
6. Test the connection
7. Save your settings

Once configured, OpenWebUI will use your CSM-1B TTS API for all text-to-speech conversion, producing high-quality speech with the selected voice.

### Using Cloned Voices with OpenWebUI

Your cloned voices will automatically appear in OpenWebUI's voice selector. Simply choose your cloned voice from the dropdown menu in the TTS settings or chat interface.

## Advanced Usage

### Conversational Context

For more natural-sounding speech in a conversation, you can use the conversation endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/audio/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nice to meet you too!",
    "speaker_id": 0,
    "context": [
      {
        "speaker": 1,
        "text": "Hello, nice to meet you.",
        "audio": "BASE64_ENCODED_AUDIO"
      }
    ]
  }' \
  --output response.wav
```

This allows the model to take into account the previous utterances for more contextually appropriate speech.

### Model Parameters

For fine-grained control, you can adjust:

- `temperature` (0.0-1.0): Higher values produce more variation but may be less stable
- `topk` (1-100): Controls diversity of generated speech
- `max_audio_length_ms`: Maximum length of generated audio in milliseconds
- `voice_consistency` (0.0-1.0): How strongly to maintain voice characteristics across segments

## Troubleshooting

### API Returns 503 Service Unavailable

- Verify your Hugging Face token has access to `sesame/csm-1b`
- Check if the model downloaded successfully in the logs
- Ensure you have enough GPU memory (at least 8GB recommended)

### Audio Quality Issues

- Try different voices - some may work better for your specific text
- Adjust temperature (lower for more stable output)
- For longer texts, the API automatically splits into smaller chunks for better quality
- For cloned voices, try recording a cleaner audio sample

### Voice Cloning Issues

- **Poor Voice Quality**: Try recording in a quieter environment with less background noise
- **Inconsistent Voice**: Provide a longer and more varied audio sample (2-3 minutes)
- **Accent Issues**: Make sure your sample contains similar words/sounds to what you'll be generating
- **Low Volume**: The sample is normalized automatically, but ensure it's not too quiet or distorted

### Voice Inconsistency

- The API maintains voice consistency across separate requests
- However, very long pauses between requests may result in voice drift
- For critical applications, consider using the same seed audio

## License

This project is released under the MIT License. The CSM-1B model is subject to its own license terms defined by Sesame.

## Acknowledgments

- [Sesame](https://www.sesame.com) for releasing the CSM-1B model
- This project is not affiliated with or endorsed by Sesame or OpenAI

---

Happy speech generating!
