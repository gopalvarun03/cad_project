from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .layers.attention import MultiheadAttention
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask_svg, _get_key_padding_mask_svg


class SVGEmbedding(nn.Module):
    """Embedding: view embed + command embed + parameter embed + positional embed"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        """concatenation-based"""
        # 3x or 4x
        if cfg.input_option == "3x" or cfg.input_option == "4x":
            self.view_embed = nn.Embedding(4, 4)
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 8)
        # 1x: keep dimension constant with other input option
        if cfg.input_option == "1x":
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 12)

        args_dim = cfg.args_dim + 1
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * cfg.svg_n_args, 128)
        self.mlp = nn.Linear(4 + 8 + 128, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)
        
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        S, N = command.shape

        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))

        """concatenation-based"""
        # 1x
        if S == 100:
            src = torch.cat([command_embedding, args_embedding], dim=-1)
        # 3x or 4x
        if S > 100:
            view_embedding = self.view_embed(view.long())
            src = torch.cat([view_embedding, command_embedding, args_embedding], dim=-1)
        
        src = self.mlp(src)
        src = self.pos_encoding(src)

        return src
    
class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)
        self.seq_len = cfg.cad_max_total_len

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        view_num = int(cfg.input_option[0])
        seq_len = view_num * cfg.svg_max_total_len
        self.embedding = SVGEmbedding(cfg, seq_len)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        padding_mask, key_padding_mask = _get_padding_mask_svg(command, seq_dim=0), _get_key_padding_mask_svg(command, seq_dim=0)
    
        src = self.embedding(view, command, args)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z

class CommandFCN(nn.Module):
    def __init__(self, d_model, n_commands):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_commands)
        )
    def forward(self, out):
        command_logits = self.mlp(out)  # Shape [S, N, n_commands]

        return command_logits

class ArgsFCN(nn.Module):
    def __init__(self, d_model, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_args * args_dim)
        )
    def forward(self, out):
        S, N, _ = out.shape

        args_logits = self.mlp(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return args_logits

class CommandDecoder(nn.Module):
    def __init__(self, cfg):
        super(CommandDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn = CommandFCN(cfg.d_model, cfg.cad_n_commands)
        
        # Sliding window attention for history
        self.window_size = cfg.history_window_size
        self.history_attention = MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.history_norm = LayerNorm(cfg.d_model)

    def forward(self, z, prev_commands=None, prev_args=None):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        
        # Apply sliding window attention to history
        if prev_commands is not None and prev_args is not None:
            S, N, D = out.shape
            out_with_history = []
            
            for i in range(S):
                # Get sliding window of history
                window_start = max(0, i - self.window_size)
                
                if i > 0:
                    # Concatenate command and args history in the window
                    window_cmd = prev_commands[window_start:i].transpose(0, 1)  # (N, window_len, D)
                    window_args = prev_args[window_start:i].transpose(0, 1)  # (N, window_len, D)
                    window_history = torch.cat([window_cmd, window_args], dim=1)  # (N, 2*window_len, D)
                    window_history = window_history.transpose(0, 1)  # (2*window_len, N, D)
                    
                    # Cross-attention to history
                    out_i = out[i:i+1]  # (1, N, D)
                    context, _ = self.history_attention(out_i, window_history, window_history)
                    out_i = self.history_norm(out_i + context)
                    out_with_history.append(out_i)
                else:
                    out_with_history.append(out[i:i+1])
            
            out = torch.cat(out_with_history, dim=0)

        command_logits = self.fcn(out)

        # guidance
        return command_logits, out

class ArgsDecoder(nn.Module):
    def __init__(self, cfg):
        super(ArgsDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)
        
        # Sliding window attention for history
        self.window_size = cfg.history_window_size
        self.history_attention = MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.history_norm = LayerNorm(cfg.d_model)

    def forward(self, z, guidance, prev_commands=None, prev_args=None):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        # guidance
        out = out + guidance
        
        # Apply sliding window attention to history
        if prev_commands is not None and prev_args is not None:
            S, N, D = out.shape
            out_with_history = []
            
            for i in range(S):
                # Get sliding window of history (including current command)
                window_start = max(0, i - self.window_size)
                
                if i >= 0:
                    # Include previous args and current + previous commands
                    if i > 0:
                        window_cmd = prev_commands[window_start:i+1].transpose(0, 1)  # (N, window_len, D)
                        window_args_hist = prev_args[window_start:i].transpose(0, 1)  # (N, window_len-1, D)
                        window_history = torch.cat([window_cmd, window_args_hist], dim=1)  # (N, 2*window_len-1, D)
                    else:
                        window_history = prev_commands[i:i+1].transpose(0, 1)  # (N, 1, D)
                    
                    window_history = window_history.transpose(0, 1)  # (history_len, N, D)
                    
                    # Cross-attention to history
                    out_i = out[i:i+1]  # (1, N, D)
                    context, _ = self.history_attention(out_i, window_history, window_history)
                    out_i = self.history_norm(out_i + context)
                    out_with_history.append(out_i)
                else:
                    out_with_history.append(out[i:i+1])
            
            out = torch.cat(out_with_history, dim=0)

        args_logits = self.fcn(out)

        return args_logits

class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(cfg.d_model // 2, cfg.d_model))

    def forward(self, z):
        return z + self.bottleneck(z)

class SVG2CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(SVG2CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.cad_n_commands = cfg.cad_n_commands
        self.cad_n_args = cfg.cad_n_args

        self.encoder = Encoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)
        
        # Embeddings for CAD tokens (for teacher forcing and autoregressive generation)
        self.command_embed = nn.Embedding(cfg.cad_n_commands, cfg.d_model)
        self.args_embed_layer = nn.Embedding(self.args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * cfg.cad_n_args, cfg.d_model)

    def embed_commands(self, commands):
        """Embed command tokens. Input: (..., ) Output: (..., d_model)"""
        return self.command_embed(commands.long())
    
    def embed_args(self, args):
        """Embed argument tokens. Input: (..., n_args) Output: (..., d_model)"""
        shape = args.shape[:-1]  # All dims except last
        args_embedded = self.args_embed_layer((args + 1).long())  # (..., n_args, 64)
        args_flat = args_embedded.reshape(*shape, -1)  # (..., n_args * 64)
        return self.args_mlp(args_flat)  # (..., d_model)

    def forward(self, views_enc, commands_enc, args_enc, cad_commands=None, cad_args=None):
        """Forward pass with optional teacher forcing.
        
        Args:
            views_enc, commands_enc, args_enc: SVG input data
            cad_commands: Ground truth CAD commands for teacher forcing (N, S)
            cad_args: Ground truth CAD args for teacher forcing (N, S, n_args)
        """
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(views_enc, commands_enc, args_enc)

        z = self.encoder(views_enc_, commands_enc_, args_enc_)
        z = self.bottleneck(z)
        
        # Teacher forcing: use ground truth history
        if cad_commands is not None and cad_args is not None:
            # Embed ground truth tokens: (N, S) -> (S, N, D)
            gt_cmd_embeds = self.embed_commands(cad_commands).transpose(0, 1)  # (S, N, D)
            gt_args_embeds = self.embed_args(cad_args).transpose(0, 1)  # (S, N, D)
            
            command_logits, guidance = self.command_decoder(z, gt_cmd_embeds, gt_args_embeds)
            args_logits = self.args_decoder(z, guidance, gt_cmd_embeds, gt_args_embeds)
        else:
            # No teacher forcing (original parallel generation)
            command_logits, guidance = self.command_decoder(z)
            args_logits = self.args_decoder(z, guidance)
        
        command_logits = _make_batch_first(command_logits)
        args_logits = _make_batch_first(args_logits)

        res = {
            "command_logits": command_logits,
            "args_logits": args_logits
        }

        return res
    
    def generate_autoregressive(self, views_enc, commands_enc, args_enc, max_len=60, temperature=1.0):
        """Autoregressive generation for inference.
        
        Args:
            views_enc, commands_enc, args_enc: SVG input data
            max_len: Maximum sequence length to generate
            temperature: Sampling temperature
        
        Returns:
            generated_commands: (N, S) tensor of command indices
            generated_args: (N, S, n_args) tensor of argument indices
        """
        from config.macro import CAD_EOS_IDX
        
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(views_enc, commands_enc, args_enc)
        N = views_enc_.size(1)
        
        z = self.encoder(views_enc_, commands_enc_, args_enc_)
        z = self.bottleneck(z)
        
        # Initialize storage for generated tokens
        generated_cmd_embeds = []  # List of (1, N, D) tensors
        generated_args_embeds = []  # List of (1, N, D) tensors
        generated_commands = []  # List of (N,) tensors
        generated_args = []  # List of (N, n_args) tensors
        
        for i in range(max_len):
            # Prepare history embeddings
            if len(generated_cmd_embeds) > 0:
                prev_cmd_embeds = torch.cat(generated_cmd_embeds, dim=0)  # (i, N, D)
                prev_args_embeds = torch.cat(generated_args_embeds, dim=0)  # (i, N, D)
            else:
                prev_cmd_embeds = None
                prev_args_embeds = None
            
            # Generate command at position i
            cmd_logits, guidance = self.command_decoder(z, prev_cmd_embeds, prev_args_embeds)
            cmd_logits_i = cmd_logits[i] / temperature  # (N, n_commands)
            cmd_probs = torch.softmax(cmd_logits_i, dim=-1)
            cmd_i = torch.argmax(cmd_probs, dim=-1)  # (N,)
            
            # Embed the generated command
            cmd_i_embed = self.embed_commands(cmd_i).unsqueeze(0)  # (1, N, D)
            generated_cmd_embeds.append(cmd_i_embed)
            generated_commands.append(cmd_i)
            
            # Update history for args generation
            prev_cmd_embeds = torch.cat(generated_cmd_embeds, dim=0)  # (i+1, N, D)
            if len(generated_args_embeds) > 0:
                prev_args_embeds = torch.cat(generated_args_embeds, dim=0)  # (i, N, D)
            else:
                prev_args_embeds = None
            
            # Generate args at position i
            args_logits = self.args_decoder(z, guidance, prev_cmd_embeds, prev_args_embeds)
            args_logits_i = args_logits[i] / temperature  # (N, n_args, args_dim)
            args_i = torch.argmax(args_logits_i, dim=-1) - 1  # (N, n_args)
            
            # Embed the generated args
            args_i_embed = self.embed_args(args_i).unsqueeze(0)  # (1, N, D)
            generated_args_embeds.append(args_i_embed)
            generated_args.append(args_i)
            
            # Early stopping if all sequences generated EOS
            if (cmd_i == CAD_EOS_IDX).all():
                break
        
        # Stack results
        generated_commands = torch.stack(generated_commands, dim=1)  # (N, S)
        generated_args = torch.stack(generated_args, dim=1)  # (N, S, n_args)
        
        # Pad to max_len if stopped early
        if generated_commands.size(1) < max_len:
            pad_len = max_len - generated_commands.size(1)
            cmd_pad = torch.full((N, pad_len), CAD_EOS_IDX, device=generated_commands.device, dtype=generated_commands.dtype)
            args_pad = torch.full((N, pad_len, self.cad_n_args), -1, device=generated_args.device, dtype=generated_args.dtype)
            generated_commands = torch.cat([generated_commands, cmd_pad], dim=1)
            generated_args = torch.cat([generated_args, args_pad], dim=1)
        
        return generated_commands, generated_args