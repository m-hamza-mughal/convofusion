import torch
import torch.nn as nn
import torch.nn.functional as F

from convofusion.models.operator.position_encoding import PositionEmbeddingSine1D
from convofusion.config import instantiate_from_config


class AudioConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, **kwargs):
        super(AudioConvEncoder, self).__init__()
        output_size = latent_dim #[-1]
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.1, inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(0.1, inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.max_seq_len = kwargs.get("max_seq_len")
        self.fps = kwargs.get("fps")
        self.sample_rate = kwargs.get("sample_rate")
        self.hop_length = kwargs.get("hop_length")

        self.audio_max_length = int((self.max_seq_len/self.fps) * self.sample_rate // self.hop_length + 1)

    def forward(self, inputs):
        """
        inputs: (batch_size, seq_len, dim)
        """
        outputs = self.main(inputs)
        return self.out_net(outputs)


class TextAudioController(nn.Module):
    def __init__(self, cfg, out_dim):
        super(TextAudioController, self).__init__()
        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.audio_encoder = instantiate_from_config(cfg.model.audio_encoder)
        self.out_dim = out_dim
        
    
        self.text_time_proj = nn.Linear(self.text_encoder.text_max_length, 
                            out_dim ) 
        self.audio_time_proj = nn.Linear(self.audio_encoder.audio_max_length,
                            out_dim ) 
        self.out_net = nn.Linear(self.out_dim, self.out_dim)
        

    def forward(self, text, audio, person_type, return_textmap=False):
        """
        text: (batch_size, seq_len, dim)
        audio: (batch_size, seq_len, dim)
        """
        
        text_emb, text_mask, token2word_map = self.text_encoder(text, return_map=return_textmap)
        

        text_mask = ~text_mask

        audio_emb = self.audio_encoder(audio)
        audio_mask = None
        if audio_mask is not None:
            audio_masked = audio_emb * audio_mask.int().unsqueeze(-1)
        else:
            audio_masked = audio_emb
        # breakpoint()
        if person_type == "spk-ta":
            if text_mask is not None:
                text_masked = text_emb * text_mask.int().unsqueeze(-1)
            else:
                text_masked = text_emb
            text_masked = text_masked.permute(0, 2, 1) 
            
            text_masked = F.pad(text_masked, (0, self.text_encoder.text_max_length-text_masked.shape[-1]))
            # breakpoint()
            text_masked = self.text_time_proj(text_masked)
            text_masked = F.leaky_relu(text_masked).permute(0, 2, 1)

            audio_masked = self.audio_time_proj(audio_masked.permute(0, 2, 1)) #, self.audio_time_proj[:, :audio_emb.shape[1]])
            audio_masked = F.leaky_relu(audio_masked).permute(0, 2, 1)

            control = text_masked + audio_masked # additional control could be added here.

            ta_fused = self.out_net(control)
            # out = self.pos_emb(out)
            return audio_emb, text_emb, audio_mask, text_mask, token2word_map, ta_fused 
        else:
            return audio_emb, text_emb, audio_mask, text_mask, token2word_map, None


if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    dim = 80
    hidden_size = 128
    latent_dim = [1, 256]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(batch_size, seq_len, dim).to(device)
    print(inputs.shape)
    model = AudioConvEncoder(dim, hidden_size, latent_dim).to(device)
    outputs = model(inputs)
    print(outputs.shape)