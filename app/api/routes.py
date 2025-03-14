"""
API routes for the CSM-1B TTS API.
"""
import os
import io
import base64
import time
import tempfile
import logging
from typing import Dict, List, Optional, Any, Union

import torch
import torchaudio
import numpy as np
from fastapi import APIRouter, Request, Response, HTTPException, Body
from fastapi.responses import StreamingResponse

from app.api.schemas import TTSRequest, ResponseFormat, Voice
from app.models import Segment

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Mapping of response_format to MIME types
MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}

@router.post("/audio/speech", summary="Generate speech from text")
async def text_to_speech(
    request: Request,
    body: Dict[str, Any] = Body(...)
):
    """
    OpenAI compatible TTS endpoint that generates speech from text using the CSM-1B model.
    """
    # Get generator from app state
    generator = request.app.state.generator
    
    # Validate model availability
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Log the request body to debug
    logger.info(f"Received TTS request: {body}")
    
    # Extract parameters with fallbacks
    text = body.get("input", "")
    if not text:
        # Try 'text' as an alternative to 'input'
        text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'input' field in request")
    
    # Handle voice parameter
    voice_name = body.get("voice", "alloy")
    
    # Convert string voice name to speaker ID
    try:
        if voice_name in ["0", "1", "2", "3", "4", "5"]:
            # Already a numeric string
            speaker = int(voice_name)
            voice_name = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"][speaker]
        elif voice_name in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            # Convert named voice to speaker ID
            voice_map = {
                "alloy": 0, 
                "echo": 1, 
                "fable": 2, 
                "onyx": 3, 
                "nova": 4, 
                "shimmer": 5
            }
            speaker = voice_map[voice_name]
        else:
            # Check if this is a cloned voice
            cloned_voice_id = None
            if hasattr(request.app.state, "voice_cloner"):
                # Check if the voice name is a cloned voice ID
                voice_cloner = request.app.state.voice_cloner
                if voice_name in voice_cloner.cloned_voices:
                    cloned_voice_id = voice_name
                    speaker = voice_cloner.cloned_voices[voice_name].speaker_id
                    logger.info(f"Using cloned voice: {voice_name} with speaker ID: {speaker}")
                else:
                    # Check if the voice name matches a cloned voice name
                    for voice_id, voice in voice_cloner.cloned_voices.items():
                        if voice.name.lower() == voice_name.lower():
                            cloned_voice_id = voice_id
                            speaker = voice.speaker_id
                            logger.info(f"Found cloned voice by name: {voice_name} (ID: {cloned_voice_id})")
                            break
            
            # Default to speaker 0 if not found
            if 'speaker' not in locals():
                logger.warning(f"Unknown voice '{voice_name}', defaulting to speaker 0")
                speaker = 0
                voice_name = "alloy"
    except Exception as e:
        logger.error(f"Error processing voice parameter: {e}")
        speaker = 0
        voice_name = "alloy"
    
    # Handle other parameters
    response_format = body.get("response_format", "mp3")
    speed = float(body.get("speed", 1.0))
    max_audio_length_ms = float(body.get("max_audio_length_ms", 90000))
    
    # Voice consistency parameters
    voice_consistency = float(body.get("voice_consistency", 0.9))  # Higher consistency by default
    max_context_segments = int(body.get("max_context_segments", 2))  # Number of context segments to use
    prioritize_consistency = body.get("prioritize_consistency", True)
    
    # Adjust temperature based on voice consistency priority
    temperature = float(body.get("temperature", 0.8))  # Lower default for more consistency
    if prioritize_consistency:
        temperature = min(temperature, 0.7)  # Ensure lower temperature for focused generation
    
    topk = int(body.get("topk", 40))
    
    # Generate audio
    try:
        logger.info(f"Generating audio for: '{text[:100]}...' with voice={voice_name} (speaker={speaker})")
        
        # Import advanced text and voice processing 
        from app.prompt_engineering import split_into_segments, format_text_for_voice
        
        # Get appropriate context
        if cloned_voice_id and hasattr(request.app.state, "voice_cloner"):
            # Use cloned voice context
            context = request.app.state.voice_cloner.get_voice_context(cloned_voice_id)
            if context:
                logger.info(f"Using cloned voice context with {len(context)} segments")
        else:
            # Use standard voice enhancement context
            from app.voice_enhancement import get_voice_segments
            context = get_voice_segments(voice_name, generator.device)
            if context:
                logger.info(f"Using {len(context)} reference segments for voice consistency")
        
        # Check if text needs to be split into segments
        # Splitting helps generate more consistent and complete audio
        segments = split_into_segments(text, max_chars=200)
        logger.info(f"Split text into {len(segments)} segments for processing")
        
        # Process each segment with voice consistency
        all_audio_chunks = []
        
        for i, segment_text in enumerate(segments):
            # Format text with voice characteristics for consistency
            formatted_text = format_text_for_voice(
                segment_text, 
                voice_name,
                segment_index=i,
                total_segments=len(segments)
            )
            
            logger.info(f"Generating segment {i+1}/{len(segments)}: '{formatted_text[:50]}...'")
            
            # Generate audio for this segment
            audio_segment = generator.generate(
                text=formatted_text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=min(max_audio_length_ms, 15000),  # Shorter segments for better control
                temperature=temperature,
                topk=topk,
            )
            
            # Process audio for quality and consistency
            from app.voice_enhancement import process_generated_audio
            processed_segment = process_generated_audio(
                audio_segment,
                voice_name,
                generator.sample_rate,
                segment_text
            )
            
            # Add to overall audio
            all_audio_chunks.append(processed_segment)
            
            # Update context for next segment to maintain consistency
            if len(all_audio_chunks) > 0 and voice_consistency > 0.5:
                # Use the current segment as context for the next segment
                # This maintains voice consistency across segments
                context = [
                    Segment(
                        speaker=speaker,
                        text=segment_text,
                        audio=processed_segment.to(generator.device)
                    )
                ] + context[:max_context_segments-1]  # Keep limited context
        
        # Combine audio chunks
        if len(all_audio_chunks) == 1:
            audio = all_audio_chunks[0]
        else:
            # Add small silence between segments for natural pauses
            silence_samples = int(0.1 * generator.sample_rate)  # 100ms silence
            silence = torch.zeros(silence_samples, device=all_audio_chunks[0].device)
            
            # Join segments with silence
            audio_parts = []
            for i, chunk in enumerate(all_audio_chunks):
                audio_parts.append(chunk)
                if i < len(all_audio_chunks) - 1:  # Don't add silence after the last chunk
                    audio_parts.append(silence)
                    
            # Concatenate all parts
            audio = torch.cat(audio_parts)
            
            logger.info(f"Combined {len(all_audio_chunks)} audio chunks, total duration: {audio.shape[0]/generator.sample_rate:.2f}s")
        
        # Apply speed adjustment if needed (using resample)
        if speed != 1.0:
            # Calculate new sample rate based on speed
            new_sample_rate = int(generator.sample_rate * speed)
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=generator.sample_rate, 
                new_freq=new_sample_rate
            )
            # Resample back to original rate to maintain compatibility
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=new_sample_rate, 
                new_freq=generator.sample_rate
            )
            
            logger.info(f"Applied speed adjustment: {speed}x")
        
        # Create temporary file for audio conversion
        with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save to WAV first (direct format for torchaudio)
        wav_path = f"{temp_path}.wav"
        torchaudio.save(wav_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Convert to requested format using ffmpeg
        import ffmpeg
        
        if response_format == "mp3":
            # For MP3, use specific bitrate for better quality
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='mp3', audio_bitrate='128k')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "opus":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='opus')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "aac":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='aac')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "flac":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='flac')
                .run(quiet=True, overwrite_output=True)
            )
        else:  # wav
            temp_path = wav_path  # Just use the WAV file directly
            response_format = "wav"  # Ensure correct MIME type
        
        # Clean up the temporary WAV file if we created a different format
        if temp_path != wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        
        # Return audio file as response
        def iterfile():
            with open(temp_path, 'rb') as f:
                yield from f
            # Clean up temp file after streaming
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return StreamingResponse(
            iterfile(),
            media_type=MIME_TYPES.get(response_format, "application/octet-stream"),
            headers={'Content-Disposition': f'attachment; filename="speech.{response_format}"'}
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Speech generation failed: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@router.post("/audio/conversation", tags=["Conversation API"])
async def conversation_to_speech(
    request: Request,
    text: str = Body(..., description="Text to convert to speech"),
    speaker_id: int = Body(0, description="Speaker ID"),
    context: List[Dict] = Body([], description="Context segments with speaker, text, and audio path"),
):
    """
    Custom endpoint for conversational TTS using CSM-1B.
    
    This is not part of the OpenAI API but provides the unique conversational
    capability of the CSM model.
    """
    # Get generator from app state
    generator = request.app.state.generator
    
    # Validate model availability
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        segments = []
        
        # Process context if provided
        for ctx in context:
            if 'speaker' not in ctx or 'text' not in ctx or 'audio' not in ctx:
                continue
                
            # Audio should be base64-encoded
            audio_data = base64.b64decode(ctx['audio'])
            audio_file = io.BytesIO(audio_data)
            
            # Save to temporary file for torchaudio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(audio_file.read())
                temp_path = temp.name
            
            # Load audio
            audio_tensor, sample_rate = torchaudio.load(temp_path)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=generator.sample_rate
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Create segment
            segments.append(
                Segment(
                    speaker=ctx['speaker'],
                    text=ctx['text'],
                    audio=audio_tensor
                )
            )
            
        logger.info(f"Conversation request: '{text}' with {len(segments)} context segments")
        
        # Format the text for better voice consistency
        from app.prompt_engineering import format_text_for_voice
        
        # Determine voice name from speaker_id
        voice_names = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice_name = voice_names[speaker_id] if 0 <= speaker_id < len(voice_names) else "alloy"
        
        formatted_text = format_text_for_voice(text, voice_name)
        
        # Generate audio with context
        audio = generator.generate(
            text=formatted_text,
            speaker=speaker_id,
            context=segments,
            max_audio_length_ms=20000,  # 20 seconds
            temperature=0.7,  # Lower temperature for more stable output
            topk=40,
        )
        
        # Process audio for better quality
        from app.voice_enhancement import process_generated_audio
        
        processed_audio = process_generated_audio(
            audio, 
            voice_name,
            generator.sample_rate,
            text
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name
        
        # Save audio
        torchaudio.save(temp_path, processed_audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Return audio file
        def iterfile():
            with open(temp_path, 'rb') as f:
                yield from f
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        logger.info(f"Generated conversation response, duration: {processed_audio.shape[0]/generator.sample_rate:.2f}s")
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={'Content-Disposition': 'attachment; filename="speech.wav"'}
        )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Conversation speech generation failed: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Conversation speech generation failed: {str(e)}")

# Add OpenAI-compatible voice list endpoint
@router.get("/audio/voices", summary="List available voices")
async def list_voices(request: Request):
    """
    OpenAI compatible endpoint that returns a list of available voices.
    """
    # Get voice descriptions from profiles if available
    try:
        from app.voice_enhancement import VOICE_PROFILES
        voices = [
            {
                "voice_id": name,
                "name": name.capitalize(),
                "preview_url": None,
                "description": f"{profile.timbre.capitalize()} voice with {int(profile.pitch_range[0])}-{int(profile.pitch_range[1])}Hz range",
                "languages": [{"language_code": "en", "name": "English"}]
            }
            for name, profile in VOICE_PROFILES.items()
        ]
    except ImportError:
        # Fallback to basic voice descriptions
        voices = [
            {
                "voice_id": "alloy",
                "name": "Alloy",
                "preview_url": None,
                "description": "Balanced voice with natural tone",
                "languages": [{"language_code": "en", "name": "English"}]
            },
            {
                "voice_id": "echo",
                "name": "Echo",
                "preview_url": None,
                "description": "Resonant voice with deeper qualities",
                "languages": [{"language_code": "en", "name": "English"}]
            },
            {
                "voice_id": "fable",
                "name": "Fable",
                "preview_url": None,
                "description": "Brighter voice with higher pitch",
                "languages": [{"language_code": "en", "name": "English"}]
            },
            {
                "voice_id": "onyx",
                "name": "Onyx",
                "preview_url": None,
                "description": "Deep, authoritative voice",
                "languages": [{"language_code": "en", "name": "English"}]
            },
            {
                "voice_id": "nova",
                "name": "Nova",
                "preview_url": None,
                "description": "Warm, pleasant voice with medium range",
                "languages": [{"language_code": "en", "name": "English"}]
            },
            {
                "voice_id": "shimmer",
                "name": "Shimmer",
                "preview_url": None,
                "description": "Light, airy voice with higher frequencies",
                "languages": [{"language_code": "en", "name": "English"}]
            }
        ]
    
    # Add cloned voices if available
    if hasattr(request.app.state, "voice_cloner"):
        voice_cloner = request.app.state.voice_cloner
        for voice in voice_cloner.list_voices():
            voices.append({
                "voice_id": voice.id,
                "name": voice.name,
                "preview_url": f"/v1/voice-cloning/voices/{voice.id}/preview",
                "description": voice.description or f"Cloned voice: {voice.name}",
                "languages": [{"language_code": "en", "name": "English"}],
                "cloned": True
            })
    
    return {"voices": voices}

# Add OpenAI-compatible models list endpoint
@router.get("/audio/models", summary="List available audio models")
async def list_models():
    """
    OpenAI compatible endpoint that returns a list of available audio models.
    """
    models = [
        {
            "id": "csm-1b",
            "name": "CSM-1B",
            "description": "Conversational Speech Model 1B from Sesame",
            "created": 1716019200,  # March 13, 2025 (from the example)
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": True,
                "voice_cloning": hasattr(router.app, "voice_cloner"),
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        },
        {
            "id": "tts-1",
            "name": "CSM-1B (Compatibility Mode)",
            "description": "CSM-1B with OpenAI TTS-1 compatibility",
            "created": 1716019200,
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": True,
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        },
        {
            "id": "tts-1-hd",
            "name": "CSM-1B (HD Mode)",
            "description": "CSM-1B with higher quality settings",
            "created": 1716019200,
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": True,
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        }
    ]
    
    return {"data": models, "object": "list"}

# Response format options endpoint
@router.get("/audio/speech/response-formats", summary="List available response formats")
async def list_response_formats():
    """List available response formats for speech synthesis."""
    formats = [
        {"name": "mp3", "content_type": "audio/mpeg"},
        {"name": "opus", "content_type": "audio/opus"},
        {"name": "aac", "content_type": "audio/aac"},
        {"name": "flac", "content_type": "audio/flac"},
        {"name": "wav", "content_type": "audio/wav"}
    ]
    
    return {"response_formats": formats}

# Simple test endpoint
@router.get("/test", summary="Test endpoint")
async def test_endpoint():
    """Simple test endpoint that returns a successful response."""
    return {"status": "ok", "message": "API is working"}

# Debug endpoint
@router.get("/debug", summary="Debug endpoint")
async def debug_info(request: Request):
    """Get debug information about the API."""
    generator = request.app.state.generator
    
    # Basic info
    debug_info = {
        "model_loaded": generator is not None,
        "device": generator.device if generator is not None else None,
        "sample_rate": generator.sample_rate if generator is not None else None,
    }
    
    # Add voice enhancement info if available
    try:
        from app.voice_enhancement import VOICE_PROFILES
        voice_info = {}
        for name, profile in VOICE_PROFILES.items():
            voice_info[name] = {
                "pitch_range": f"{profile.pitch_range[0]}-{profile.pitch_range[1]}Hz",
                "timbre": profile.timbre,
                "ref_segments": len(profile.reference_segments),
            }
        debug_info["voice_profiles"] = voice_info
    except ImportError:
        debug_info["voice_profiles"] = "Not available"
        
    # Add voice cloning info if available
    if hasattr(request.app.state, "voice_cloner"):
        voice_cloner = request.app.state.voice_cloner
        debug_info["voice_cloning"] = {
            "enabled": True,
            "cloned_voices_count": len(voice_cloner.list_voices()),
            "cloned_voices": [v.name for v in voice_cloner.list_voices()]
        }
    else:
        debug_info["voice_cloning"] = {"enabled": False}
    
    # Add memory usage info for CUDA
    if torch.cuda.is_available():
        debug_info["cuda"] = {
            "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "max_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    
    return debug_info

# Voice diagnostics endpoint
@router.get("/debug/voices", summary="Voice diagnostics")
async def voice_diagnostics():
    """Get diagnostic information about voice references."""
    try:
        from app.voice_enhancement import VOICE_PROFILES
        
        diagnostics = {}
        for name, profile in VOICE_PROFILES.items():
            ref_info = []
            for i, ref in enumerate(profile.reference_segments):
                if ref is not None:
                    duration = ref.shape[0] / 24000  # Assume 24kHz
                    ref_info.append({
                        "index": i,
                        "duration_seconds": f"{duration:.2f}",
                        "samples": ref.shape[0],
                        "min": float(ref.min()),
                        "max": float(ref.max()),
                        "rms": float(torch.sqrt(torch.mean(ref ** 2))),
                    })
            
            diagnostics[name] = {
                "speaker_id": profile.speaker_id,
                "pitch_range": f"{profile.pitch_range[0]}-{profile.pitch_range[1]}Hz",
                "references": ref_info,
                "reference_count": len(ref_info),
            }
        
        return {"diagnostics": diagnostics}
    except ImportError:
        return {"error": "Voice enhancement module not available"}

# Specialized debugging endpoint for speech generation
@router.post("/debug/speech", summary="Debug speech generation")
async def debug_speech(
    request: Request,
    text: str = Body(..., embed=True),
    voice: str = Body("alloy", embed=True),
    use_enhancement: bool = Body(True, embed=True)
):
    """Debug endpoint for speech generation with enhancement options."""
    generator = request.app.state.generator
    
    if generator is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert voice name to speaker ID
        voice_map = {
            "alloy": 0, 
            "echo": 1, 
            "fable": 2, 
            "onyx": 3, 
            "nova": 4, 
            "shimmer": 5
        }
        speaker = voice_map.get(voice, 0)
        
        # Format text if using enhancement
        if use_enhancement:
            from app.prompt_engineering import format_text_for_voice
            formatted_text = format_text_for_voice(text, voice)
            logger.info(f"Using formatted text: {formatted_text}")
        else:
            formatted_text = text
            
        # Get context if using enhancement
        if use_enhancement:
            from app.voice_enhancement import get_voice_segments
            context = get_voice_segments(voice, generator.device)
            logger.info(f"Using {len(context)} context segments")
        else:
            context = []
            
        # Generate audio
        start_time = time.time()
        audio = generator.generate(
            text=formatted_text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=10000,  # 10 seconds
            temperature=0.7 if use_enhancement else 0.9,
            topk=40 if use_enhancement else 50,
        )
        generation_time = time.time() - start_time
        
        # Process audio if using enhancement
        if use_enhancement:
            from app.voice_enhancement import process_generated_audio
            start_time = time.time()
            processed_audio = process_generated_audio(audio, voice, generator.sample_rate, text)
            processing_time = time.time() - start_time
        else:
            processed_audio = audio
            processing_time = 0
        
        # Save to temporary WAV file
        temp_path = f"/tmp/debug_speech_{voice}_{int(time.time())}.wav"
        torchaudio.save(temp_path, processed_audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Also save original if enhanced
        if use_enhancement:
            orig_path = f"/tmp/debug_speech_{voice}_original_{int(time.time())}.wav"
            torchaudio.save(orig_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        else:
            orig_path = temp_path
            
        # Calculate audio metrics
        duration = processed_audio.shape[0] / generator.sample_rate
        rms = float(torch.sqrt(torch.mean(processed_audio ** 2)))
        peak = float(processed_audio.abs().max())
        
        return {
            "status": "success",
            "message": f"Audio generated successfully and saved to {temp_path}",
            "audio": {
                "duration_seconds": f"{duration:.2f}",
                "samples": processed_audio.shape[0],
                "sample_rate": generator.sample_rate,
                "rms_level": f"{rms:.3f}",
                "peak_level": f"{peak:.3f}",
            },
            "processing": {
                "enhancement_used": use_enhancement,
                "generation_time_seconds": f"{generation_time:.3f}",
                "processing_time_seconds": f"{processing_time:.3f}",
                "original_path": orig_path,
                "processed_path": temp_path,
            }
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Debug speech generation failed: {e}\n{error_trace}")
        return {
            "status": "error",
            "message": str(e),
            "traceback": error_trace
        }