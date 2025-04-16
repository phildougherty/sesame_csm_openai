"""Streaming support for the TTS API."""
import asyncio
import io
import logging
import time
from typing import AsyncGenerator, Optional, List
import torch
import torchaudio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import SpeechRequest, ResponseFormat
from app.prompt_engineering import split_into_segments
from app.models import Segment

logger = logging.getLogger(__name__)
router = APIRouter()

class AudioChunker:
    """Handle audio chunking for streaming responses."""
    def __init__(self,
                 sample_rate: int,
                 format: str = "mp3",
                 chunk_size_ms: int = 200):  # Smaller chunks for better streaming
        """
        Initialize audio chunker.
        Args:
            sample_rate: Audio sample rate in Hz
            format: Output audio format (mp3, opus, etc.)
            chunk_size_ms: Size of each chunk in milliseconds
        """
        self.sample_rate = sample_rate
        self.format = format.lower()
        self.chunk_size_samples = int(sample_rate * (chunk_size_ms / 1000))
        logger.info(f"Audio chunker initialized with {chunk_size_ms}ms chunks ({self.chunk_size_samples} samples)")

    async def chunk_audio(self,
                    audio: torch.Tensor,
                    delay_ms: int = 0) -> AsyncGenerator[bytes, None]:
        """
        Convert audio tensor to streaming chunks.
        Args:
            audio: Audio tensor to stream
            delay_ms: Artificial delay between chunks (for testing)
        Yields:
            Audio chunks as bytes
        """
        # Ensure audio is on CPU
        if audio.is_cuda:
            audio = audio.cpu()
        # Calculate number of chunks
        num_samples = audio.shape[0]
        num_chunks = (num_samples + self.chunk_size_samples - 1) // self.chunk_size_samples
        logger.info(f"Streaming {num_samples} samples as {num_chunks} chunks")
        for i in range(num_chunks):
            start_idx = i * self.chunk_size_samples
            end_idx = min(start_idx + self.chunk_size_samples, num_samples)
            # Extract chunk
            chunk = audio[start_idx:end_idx]
            # Convert to bytes in requested format
            chunk_bytes = await self._format_chunk(chunk)
            # Add artificial delay if requested (for testing)
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)
            yield chunk_bytes

    async def _format_chunk(self, chunk: torch.Tensor) -> bytes:
        """Convert audio chunk to bytes in the specified format."""
        buf = io.BytesIO()
        # Ensure chunk is 1D and on CPU
        if len(chunk.shape) == 1:
            chunk = chunk.unsqueeze(0)  # Add channel dimension
        # Ensure chunk is on CPU
        if chunk.is_cuda:
            chunk = chunk.cpu()
        # Save to buffer in specified format
        if self.format == "mp3":
            torchaudio.save(buf, chunk, self.sample_rate, format="mp3")
        elif self.format == "opus":
            torchaudio.save(buf, chunk, self.sample_rate, format="opus")
        elif self.format == "aac":
            torchaudio.save(buf, chunk, self.sample_rate, format="aac")
        elif self.format == "flac":
            torchaudio.save(buf, chunk, self.sample_rate, format="flac")
        elif self.format == "wav":
            torchaudio.save(buf, chunk, self.sample_rate, format="wav")
        else:
            # Default to mp3
            torchaudio.save(buf, chunk, self.sample_rate, format="mp3")
        # Get bytes from buffer
        buf.seek(0)
        return buf.read()

# Helper function to get speaker ID for a voice
def get_speaker_id(app_state, voice):
    """Helper function to get speaker ID from voice name or ID"""
    if hasattr(app_state, "voice_speaker_map") and voice in app_state.voice_speaker_map:
        return app_state.voice_speaker_map[voice]
    # Standard voices mapping
    voice_to_speaker = {"alloy": 0, "echo": 1, "fable": 2, "onyx": 3, "nova": 4, "shimmer": 5}
    if voice in voice_to_speaker:
        return voice_to_speaker[voice]
    # Try parsing as integer
    try:
        speaker_id = int(voice)
        if 0 <= speaker_id < 6:
            return speaker_id
    except (ValueError, TypeError):
        pass
    # Check cloned voices if the voice cloner exists
    if hasattr(app_state, "voice_cloner") and app_state.voice_cloner is not None:
        # Check by ID
        if voice in app_state.voice_cloner.cloned_voices:
            return app_state.voice_cloner.cloned_voices[voice].speaker_id
        # Check by name
        for v_id, v_info in app_state.voice_cloner.cloned_voices.items():
            if v_info.name.lower() == voice.lower():
                return v_info.speaker_id
    # Default to alloy
    return 0

@router.post("/audio/speech/stream", tags=["Audio"])
async def stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio of text being spoken by a realistic voice.
    This endpoint provides a streaming interface for TTS.
    """
    # Check if model is loaded
    if not hasattr(request.app.state, "generator") or request.app.state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    # Get request parameters
    model = speech_request.model
    input_text = speech_request.input
    voice = speech_request.voice
    response_format = speech_request.response_format
    speed = speech_request.speed
    temperature = speech_request.temperature
    max_audio_length_ms = speech_request.max_audio_length_ms

    # Log the request
    logger.info(f"Real-time streaming speech from text ({len(input_text)} chars) with voice '{voice}'")

    # Check if text is empty
    if not input_text or len(input_text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    # Get speaker ID for the voice
    speaker_id = get_speaker_id(request.app.state, voice)
    if speaker_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available voices: {request.app.state.available_voices}"
        )

    try:
        # Create media type based on format
        media_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
        }.get(response_format, "audio/mpeg")

        # Create the chunker for streaming
        sample_rate = request.app.state.sample_rate
        chunker = AudioChunker(sample_rate, response_format)

        # Split text into segments using the imported function
        from app.prompt_engineering import split_into_segments
        text_segments = split_into_segments(input_text, max_chars=50)  # Smaller segments for faster first response
        
        logger.info(f"Split text into {len(text_segments)} segments for incremental streaming")

        async def generate_streaming_audio():
            # Check for cloned voice
            voice_info = None
            from_cloned_voice = False
            
            if hasattr(request.app.state, "voice_cloning_enabled") and request.app.state.voice_cloning_enabled:
                if hasattr(request.app.state, "get_voice_info"):
                    voice_info = request.app.state.get_voice_info(voice)
                    from_cloned_voice = voice_info and voice_info.get("type") == "cloned"
                    if from_cloned_voice:
                        # Use cloned voice context for first segment
                        voice_cloner = request.app.state.voice_cloner
                        context = voice_cloner.get_voice_context(voice_info["voice_id"])
                    else:
                        # Use standard voice context
                        from app.voice_enhancement import get_voice_segments
                        context = get_voice_segments(voice, request.app.state.device)
                else:
                    # Use standard voice context
                    from app.voice_enhancement import get_voice_segments
                    context = get_voice_segments(voice, request.app.state.device)
            else:
                # Use standard voice context
                from app.voice_enhancement import get_voice_segments
                context = get_voice_segments(voice, request.app.state.device)

            # Send an empty chunk to initialize the connection
            yield b''

            # Process each text segment incrementally and stream in real time
            for i, segment_text in enumerate(text_segments):
                try:
                    logger.info(f"Generating segment {i+1}/{len(text_segments)}")
                    
                    # Generate audio for this segment - use async to avoid blocking
                    if from_cloned_voice:
                        # Generate with cloned voice
                        voice_cloner = request.app.state.voice_cloner
                        
                        # Convert to asynchronous with asyncio.to_thread
                        segment_audio = await asyncio.to_thread(
                            voice_cloner.generate_speech,
                            segment_text,
                            voice_info["voice_id"],
                            temperature=temperature,
                            topk=30,
                            max_audio_length_ms=2000  # Keep segments short for streaming
                        )
                    else:
                        # Use standard voice with generator
                        segment_audio = await asyncio.to_thread(
                            request.app.state.generator.generate,
                            segment_text,
                            speaker_id,
                            context,
                            max_audio_length_ms=2000,  # Short for quicker generation
                            temperature=temperature
                        )
                    
                    # Process audio quality for this segment if enhancement enabled
                    if hasattr(request.app.state, "voice_enhancement_enabled") and request.app.state.voice_enhancement_enabled:
                        try:
                            from app.voice_enhancement import process_generated_audio
                            segment_audio = process_generated_audio(
                                audio=segment_audio,
                                voice_name=voice,
                                sample_rate=sample_rate,
                                text=segment_text
                            )
                        except Exception as enhance_error:
                            logger.warning(f"Voice enhancement failed: {enhance_error}. Using unenhanced audio.")
                        
                    # Handle speed adjustment
                    if speed != 1.0 and speed > 0:
                        try:
                            # Adjust speed using torchaudio
                            effects = [
                                ["tempo", str(speed)]
                            ]
                            audio_cpu = segment_audio.cpu()
                            adjusted_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                                audio_cpu.unsqueeze(0), 
                                sample_rate, 
                                effects
                            )
                            segment_audio = adjusted_audio.squeeze(0)
                            logger.info(f"Adjusted speech speed to {speed}x")
                        except Exception as e:
                            logger.warning(f"Failed to adjust speech speed: {e}")
                    
                    # Stream the segment in chunks
                    async for chunk in chunker.chunk_audio(segment_audio):
                        yield chunk
                    
                    # Update context with this segment for next generation
                    context = [
                        Segment(
                            text=segment_text,
                            speaker=speaker_id,
                            audio=segment_audio
                        )
                    ]
                
                except Exception as segment_error:
                    logger.error(f"Error generating segment {i+1}: {segment_error}")
                    # Continue to next segment
            
        # Return streaming response
        return StreamingResponse(
            generate_streaming_audio(),
            media_type=media_type,
            headers={
                "X-Accel-Buffering": "no",  # Prevent buffering in nginx
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in stream_speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/speech/streaming", tags=["Audio"])
async def openai_stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio in OpenAI-compatible streaming format.
    This endpoint is compatible with the OpenAI streaming TTS API.
    """
    # Use the same logic as the stream_speech endpoint but with a different name
    # to maintain the OpenAI API naming convention
    return await stream_speech(request, speech_request)

# Additional route for OpenAI compatibility
@router.post("/v1/audio/speech/stream", tags=["Audio"])
async def v1_stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio of text being spoken (v1 prefix route).
    This endpoint is the v1 version of the streaming API.
    """
    return await stream_speech(request, speech_request)

# Additional route for OpenAI compatibility 
@router.post("/v1/audio/speech/streaming", tags=["Audio"])
async def v1_openai_stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio in OpenAI-compatible streaming format (v1 prefix route).
    This endpoint is the v1 version of the OpenAI streaming API.
    """
    return await stream_speech(request, speech_request)