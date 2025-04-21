"""Torchtune models for CSM-1B."""
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn

# Set up logging
logger = logging.getLogger(__name__)

# First, try to import llama3_2 from torchtune directly
try:
    import torchtune
    logger.info(f"Torchtune version: {getattr(torchtune, '__version__', 'unknown')}")
    
    # Print available modules in torchtune.models
    try:
        import torchtune.models
        logger.info(f"Available modules in torchtune.models: {dir(torchtune.models)}")
    except Exception as e:
        logger.error(f"Error inspecting torchtune.models: {e}")
    
    # Try to import llama3_2 model
    try:
        from torchtune.models.llama3_2 import llama3_2
        logger.info("Successfully imported llama3_2 from torchtune")
    except ImportError as e:
        logger.warning(f"Could not import llama3_2: {e}")
        # Try to import regular llama as fallback
        try:
            from torchtune.models.llama import llama
            logger.info("Using llama from torchtune.models.llama as fallback")
            llama3_2 = llama  # Alias llama as llama3_2
        except ImportError:
            logger.error("Could not import llama model either. Will use custom implementation.")
            llama3_2 = None
except ImportError as e:
    logger.error(f"Torchtune not available: {e}")
    torchtune = None
    llama3_2 = None


# Define our own model implementations as fallbacks
def llama3_2_1B_custom():
    """Create a Llama 3.2 1B model."""
    from app.custom_transformer import CustomTransformerDecoder
    return CustomTransformerDecoder(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def llama3_2_100M_custom():
    """Create a Llama 3.2 100M model."""
    from app.custom_transformer import CustomTransformerDecoder
    return CustomTransformerDecoder(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


# Setup fallback to our own implementations if needed
if llama3_2 is None:
    logger.warning("Using custom implementations for Llama models")
    FLAVORS = {
        "llama-1B": llama3_2_1B_custom,
        "llama-100M": llama3_2_100M_custom,
    }
else:
    logger.info("Using torchtune implementations for Llama models")
    FLAVORS = {
        "llama-1B": lambda: llama3_2(
            vocab_size=128_256,
            num_layers=16,
            num_heads=32,
            num_kv_heads=8,
            embed_dim=2048,
            max_seq_len=2048,
            intermediate_dim=8192,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500_000,
            scale_factor=32,
        ),
        "llama-100M": lambda: llama3_2(
            vocab_size=128_256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            embed_dim=1024,
            max_seq_len=2048,
            intermediate_dim=8192,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500_000,
            scale_factor=32,
        ),
    }


def _prepare_transformer(model):
    """Prepare transformer for use."""
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """Create causal mask."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """Index causal mask.
    
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)
    
    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    """Do multinomial sampling without a cuda synchronization."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """Sample from top-k logits."""
    logits = logits / temperature
    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    """Model arguments."""
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    """CSM-1B model."""
    
    def __init__(self, args: ModelArgs):
        """Initialize model."""
        super().__init__()
        self.args = args
        logger.info(f"Creating model with backbone: {args.backbone_flavor}, decoder: {args.decoder_flavor}")
        
        # Load backbone and decoder
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())
        self.backbone = torch.compile(self.backbone, mode='max-autotune', fullgraph=True, backend='inductor')
        self.decoder = torch.compile(self.decoder, mode='max-autotune', fullgraph=True, backend='inductor')
        
        # Embeddings
        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)
        
        # Projection and heads
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))
        
        # Initialize audio head
        nn.init.normal_(self.audio_head, mean=0.0, std=0.02)
    
    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)
        
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device))
    
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """Generate a frame of audio tokens.
        
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
        
        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s = tokens.size()[:2]
        
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        
        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        
        return curr_sample
    
    def reset_caches(self):
        """Reset KV caches."""
        self.backbone.reset_caches()
        self.decoder.reset_caches()
    
    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens."""
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)
    
    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens."""
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )
        
        return torch.cat([audio_embeds, text_embeds], dim=-2)