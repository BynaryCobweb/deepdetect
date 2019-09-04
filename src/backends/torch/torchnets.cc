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

#include "torchnets.h"

using namespace torch;

namespace dd {
    MultiheadAttentionImpl::MultiheadAttentionImpl(const TransformerOptions &config) 
        : _head_count{config.head_count}, 
          _head_size{config.hidden_size / config.head_count}
    {

        int all_head_size = _head_size * _head_count;
        
        _values = register_module("values", nn::Linear(config.hidden_size, all_head_size));
        _keys = register_module("keys", nn::Linear(config.hidden_size, all_head_size));
        _query = register_module("query", nn::Linear(config.hidden_size, all_head_size));
        _output = register_module("output", nn::Linear(all_head_size, config.hidden_size));
    }

    Tensor MultiheadAttentionImpl::get_attention_probs(Tensor hidden_layer, Tensor attention_mask)
    {
        auto q = separate_heads(_query(hidden_layer));
        auto k = separate_heads(_keys(hidden_layer));

        auto scores = q.matmul(k.transpose(-1, -2)) / sqrt(_head_size);
        // TODO check this line
        // scores += attention_mask;
        auto probs = softmax(scores, -1);
        return probs;
    }

    Tensor MultiheadAttentionImpl::forward(Tensor hidden_layer, Tensor attention_mask)
    {
        auto v = separate_heads(_values(hidden_layer));
        auto probs = get_attention_probs(hidden_layer, attention_mask);

        auto multihead_output = probs.matmul(v);
        multihead_output = join_heads_back(multihead_output);

        return _output(multihead_output);
    }

    Tensor MultiheadAttentionImpl::separate_heads(Tensor t)
    {
        // from tensor: n_batch * n_words * all_head_size
        // to tensor: n_batch * head_count * n_words * head_size
        auto src_shape = t.sizes();
        std::vector<int64_t> shape{src_shape.begin(), src_shape.end() - 1};
        shape.push_back(_head_count);
        shape.push_back(_head_size);

        t = t.view(IntList(shape));
        return t.permute(IntList{0, 2, 1, 3});
    }

    Tensor MultiheadAttentionImpl::join_heads_back(Tensor t)
    {
        // from tensor: n_batch * head_count * n_words * head_size
        // to tensor: n_batch * n_words * all_head_size
        t = t.permute(IntList{0, 2, 1, 3}).contiguous();

        auto src_shape = t.sizes();
        std::vector<int64_t> shape{src_shape.begin(), src_shape.end() - 2};
        shape.push_back(_head_count * _head_size);
        return t.view(IntList(shape));
    }


    TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(const TransformerOptions &config)
        : _attention{config}, _dropout{config.dropout}
    {
        _linear1 = register_module("linear1", nn::Linear(config.hidden_size, config.intermediate_size));
        _linear2 = register_module("linear2", nn::Linear(config.intermediate_size, config.hidden_size));
        _norm1 = register_module("norm1", nn::BatchNorm(config.hidden_size));
        _norm2 = register_module("norm2", nn::BatchNorm(config.hidden_size));
        register_module("dropout", _dropout);
    }

    Tensor TransformerEncoderLayerImpl::forward(Tensor hidden_layer, Tensor attention_mask) 
    {
        Tensor att_output = _attention(hidden_layer, attention_mask);
        Tensor output = _norm1(_linear1(att_output));
        output = _dropout(output);
        // TODO original model use some residual architecture here
        output = _norm2(_linear2(output));
        return output;
    }


    TransformerEncoderImpl::TransformerEncoderImpl(const TransformerOptions &config)
    {
        for (int i = 0; i < config.num_layers; ++i)
        {
            _layers.emplace_back(config);
        }
    }

    Tensor TransformerEncoderImpl::forward(Tensor hidden_layer, Tensor attention_mask)
    {
        for (auto &layer : _layers)
        {
            hidden_layer = layer(hidden_layer, attention_mask);
        }
        return hidden_layer;
    }
}