/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef TORCHNETS_H
#define TORCHNETS_H

#include <torch/torch.h>

namespace dd {

struct TransformerOptions {
    int hidden_size;
    int intermediate_size;
    int head_count;
    int num_layers;
    double dropout = 0;
};


struct MultiheadAttentionImpl : torch::nn::Module {
    MultiheadAttentionImpl(const TransformerOptions &config);

    torch::Tensor get_attention_probs(torch::Tensor hidden_layer, torch::Tensor attention_mask);

    torch::Tensor forward(torch::Tensor hidden_layer, torch::Tensor attention_mask);
private:
    int _head_count;
    int _head_size;

    torch::nn::Linear _values = nullptr, _keys = nullptr, _query = nullptr, _output = nullptr;


    torch::Tensor separate_heads(torch::Tensor t);

    torch::Tensor join_heads_back(torch::Tensor t);
};
TORCH_MODULE(MultiheadAttention);


struct TransformerEncoderLayerImpl : torch::nn::Module {
    TransformerEncoderLayerImpl(const TransformerOptions &config);

    torch::Tensor forward(torch::Tensor hidden_layer, torch::Tensor attention_mask);
private:
    MultiheadAttention _attention;
    torch::nn::Linear _linear1 = nullptr, _linear2 = nullptr;
    torch::nn::Dropout _dropout;
    torch::nn::BatchNorm _norm1 = nullptr, _norm2 = nullptr;
};
TORCH_MODULE(TransformerEncoderLayer);


struct TransformerEncoderImpl : torch::nn::Module {
    TransformerEncoderImpl(const TransformerOptions &config);

    torch::Tensor forward(torch::Tensor hidden_layer, torch::Tensor attention_mask);

private:
    std::vector<TransformerEncoderLayer> _layers;
};
TORCH_MODULE(TransformerEncoder);

} // namespace dd

#endif // TORCHNETS_H
