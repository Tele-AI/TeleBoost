# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
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
from unittest.mock import Mock, patch

import torch
import torch.nn as nn


@patch("teleboost.engines.teletron.get_args")
def get_model(mock_teleboost):
    from teleboost.engines.teletron.transformer import TransformerGeneralMixin

    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 2
    mock_teleboost.return_value = args

    class TestConfig:
        hidden_size = 1024
        num_layers = 4
        recompute_method = "block"
        recompute_granularity = "full"
        recompute_num_layers = args.recompute_num_layers

    class TestModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_layers)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    class TestRecomputeModel(TransformerGeneralMixin, TestModel):
        def __init__(self, config):
            super().__init__(config)
            self.enable_activation_checkpointing(self.blocks)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    test_config = TestConfig()
    model = TestModel(test_config)
    recompute_model = TestRecomputeModel(test_config)

    recompute_model.load_state_dict(model.state_dict())

    return test_config, model, recompute_model


def is_close_by_normalized_euclid_dist(output, parallel_output):
    wan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (wan_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True
    else:
        return False


def test_block_recompute():
    config, model, recompute_model = get_model()
    x = torch.rand(16, 1024)

    # forward compare
    output = model(x)
    output.backward(torch.ones_like(output))
    # No with_stack: the assertion below only counts op invocations, and stack
    # capture makes kineto decode host python frames — which crashes with
    # UnicodeDecodeError when any frame name is non-UTF-8 (seen under pytest
    # on macOS with CJK locales).
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        recompute_output = recompute_model(x)
        recompute_output.backward(torch.ones_like(recompute_output))

    if torch.all(output == recompute_output):
        forward_result = "success"
    else:
        forward_result = "fail"

    # backward grad compare
    model_grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    parallel_moedl_grads = {name: param.grad for name, param in recompute_model.named_parameters() if param.grad is not None}
    for name in model_grads:
        if is_close_by_normalized_euclid_dist(model_grads[name], parallel_moedl_grads[name]):
            backward_result = "success"
        else:
            backward_result = "fail"

    # backward recompute check
    linear_events = [e for e in prof.key_averages() if e.key.endswith("linear")]
    real_linear_calls = sum(e.count for e in linear_events)
    total_linear_calls = config.num_layers + min(config.num_layers, config.recompute_num_layers)

    assert forward_result == "success"
    assert backward_result == "success"
    assert real_linear_calls == total_linear_calls
