from typing import Any

import torch
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder


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
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        notes: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        context = self.encoder(inputs)
        loss = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            notes=notes,
            context=context,
            mask=mask,
            **kwargs,
        )
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> list[EncodedSymbol]:
        start_token = (torch.LongTensor([self.config.bos_token] * len(x))[:, None]).to(x.device)
        nonote_token = (torch.LongTensor([self.config.nonote_token] * len(x))[:, None]).to(x.device)

        context = self.encoder(x)
        out = self.decoder.generate(start_token, nonote_token, context=context)

        return out
