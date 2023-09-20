# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import logging

class ConvASRDecoder(torch.nn.Module):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
        https://arxiv.org/pdf/2005.04290.pdf
    """

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super().__init__()

        if vocabulary is None and num_classes < 0:
            raise ValueError(
                f"Neither of the vocabulary and num_classes are set! At least one of them need to be set."
            )

        if num_classes <= 0:
            num_classes = len(vocabulary)
            logging.info(f"num_classes of ConvASRDecoder is set to the size of the vocabulary: {num_classes}.")

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary

        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        # self.apply(lambda x: init_weights(x, mode=init_mode))

        # accepted_adapters = [adapter_utils.LINEAR_ADAPTER_CLASSPATH]
        # self.set_accepted_adapter_types(accepted_adapters)

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(self, encoder_output):
        # Adapter module forward step
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)  # [B, C, T]

        if self.temperature != 1.0:
            return torch.nn.functional.log_softmax(
                self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature, dim=-1
            )
        return torch.nn.functional.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])
