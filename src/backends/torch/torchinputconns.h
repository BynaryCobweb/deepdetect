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
#include "csvtsinputfileconn.h"

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


            for (const cv::Mat &bgr : this->_images) {
                _height = bgr.rows;
                _width = bgr.cols;

                std::vector<int64_t> sizes{ 1, _height, _width, 3 };
                at::TensorOptions options(at::ScalarType::Byte);
                at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
                imgt = imgt.toType(at::kFloat).mul(1./255.).permute({0, 3, 1, 2});

                // bgr to rgb
                at::Tensor indexes = torch::ones(3, at::kLong);
                indexes[0] = 2;
                indexes[2] = 0;
                imgt = torch::index_select(imgt, 1, indexes);

                _in_tensors.push_back(imgt);
            }
        }

    public:
        std::vector<at::Tensor> _in_tensors;
    };
} // namespace dd

#endif // TORCHINPUTCONNS_H