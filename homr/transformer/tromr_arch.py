from typing import Any

import torch
from torch import nn

from homr.debug import AttentionDebug
from homr.transformer.configs import Config
from homr.transformer.decoder import get_decoder
from homr.transformer.encoder import get_encoder


class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def eval_mode(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms_seq: torch.Tensor,
        pitchs_seq: torch.Tensor,
        lifts_seq: torch.Tensor,
        note_seq: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        encoded = self.encoder(inputs)
        loss = self.decoder(
            rhythms_seq, pitchs_seq, lifts_seq, note_seq, context=encoded, mask=mask, **kwargs
        )
        return loss

    @torch.no_grad()
    def generate(
        self, x: torch.Tensor, keep_all_symbols_in_chord: bool, debug: AttentionDebug | None
    ) -> list[str]:
        start_token = (torch.LongTensor([self.config.bos_token] * len(x))[:, None]).to(x.device)
        nonote_token = (torch.LongTensor([self.config.nonote_token] * len(x))[:, None]).to(x.device)

        context = self.encoder(x)
        return self.decoder.generate(
            start_token,
            nonote_token,
            self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
            keep_all_symbols_in_chord=keep_all_symbols_in_chord,
            debug=debug,
        )
