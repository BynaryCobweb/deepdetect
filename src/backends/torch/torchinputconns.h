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

#ifndef TORCHINPUTCONNS_H
#define TORCHINPUTCONNS_H

#include "imginputfileconn.h"
#include "txtinputfileconn.h"

#include <vector>

namespace dd
{
    class TorchInputInterface
    {
    public:
        TorchInputInterface() {}
        TorchInputInterface(const TorchInputInterface &i)
        {

        }

        ~TorchInputInterface() {}

        torch::Tensor toLongTensor(std::vector<int64_t> &values) {
            int64_t val_size = values.size();
            return torch::from_blob(&values[0], at::IntList{1, val_size}, at::kLong);
        }

        at::Tensor _in;
        at::Tensor _attention_mask;
    };

    class ImgTorchInputFileConn : public ImgInputFileConn, public TorchInputInterface
    {
    public:
        ImgTorchInputFileConn()
            :ImgInputFileConn() {}
        ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
            :ImgInputFileConn(i),TorchInputInterface(i) {}
        ~ImgTorchInputFileConn() {}

        // for API info only
        int width() const
        {
            return _width;
        }

        // for API info only
        int height() const
        {
            return _height;
        }

        void init(const APIData &ad)
        {
            ImgInputFileConn::init(ad);
        }
        
        void transform(const APIData &ad)
        {
            try
            {
                ImgInputFileConn::transform(ad);
            }
            catch(const std::exception& e)
            {
                throw;
            }

            std::vector<at::Tensor> tensors;

            for (const cv::Mat &bgr : this->_images) {
                _height = bgr.rows;
                _width = bgr.cols;

                std::vector<int64_t> sizes{ _height, _width, 3 };
                at::TensorOptions options(at::ScalarType::Byte);
                at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
                imgt = imgt.toType(at::kFloat).mul(1./255.).permute({2, 0, 1});

                // bgr to rgb
                at::Tensor indexes = torch::ones(3, at::kLong);
                indexes[0] = 2;
                indexes[2] = 0;
                imgt = torch::index_select(imgt, 0, indexes);

                tensors.push_back(imgt);
            }

            _in = torch::stack(tensors, 0);
        }

    public:
    };


    class TxtTorchInputFileConn : public TxtInputFileConn, public TorchInputInterface
    {
    public:
        TxtTorchInputFileConn()
            : TxtInputFileConn() {}
        TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
            : TxtInputFileConn(i), TorchInputInterface(i) {}
        ~TxtTorchInputFileConn() {}

        // for API info only
        int width() const
        {
            return _width;
        }

        // for API info only
        int height() const
        {
            return _height;
        }

        void transform(const APIData &ad)
        {
            try
            {
                TxtInputFileConn::transform(ad);
            }
            catch(const std::exception& e)
            {
                throw;
            }

            std::vector<int64_t> input_ids{
                101, 2489, 4443, 1999, 1016, 1037, 1059, 2243, 2135, 4012, 2361,
                2000, 2663, 6904, 2452, 2345, 1056, 25509, 2015, 7398, 2089, 2384,
                1012, 3793, 6904, 2000, 6584, 12521, 2487, 2000, 4374, 4443, 3160,
                1006, 2358, 2094, 19067, 2102, 3446, 1007, 1056, 1004, 1039, 1005,
                1055, 6611, 5511, 19961, 22407, 18613, 23352, 7840, 15136, 1005, 1055, 102
            };

            at::Tensor input_ids_tensor = toLongTensor(input_ids);
            at::Tensor input_mask_tensor = torch::ones_like(input_ids_tensor);
            // at::Tensor token_type_ids_tensor = torch::zeros_like(input_ids_tensor);
            
            int64_t padding_size = _input_size - input_ids_tensor.sizes().back();
            input_ids_tensor = torch::constant_pad_nd(
                input_ids_tensor, at::IntList{0, padding_size}, 0);
            input_mask_tensor = torch::constant_pad_nd(
                input_mask_tensor, at::IntList{0, padding_size}, 0);
            // token_type_ids_tensor = torch::constant_pad_nd(
            //    token_type_ids_tensor, at::IntList{0, padding_size}, 0);
            
            _in = input_ids_tensor;
            _attention_mask = input_mask_tensor;
        }

    public:
        int _width = 0;
        int _height = 0;
        int64_t _input_size = 500;
    };
} // namespace dd

#endif // TORCHINPUTCONNS_H