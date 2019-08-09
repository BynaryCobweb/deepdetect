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

#include <vector>

#include <torch/torch.h>

#include "imginputfileconn.h"
#include "txtinputfileconn.h"

namespace dd
{
    typedef torch::data::Example<std::vector<at::Tensor>, std::vector<at::Tensor>> TorchBatch;

    class TorchDataset : public torch::data::BatchDataset
        <TorchDataset, c10::optional<TorchBatch>>
    {
    private:
        bool _shuffle = false;
        long _seed = -1;
        std::vector<int64_t> _indices;

    public:
        std::vector<TorchBatch> _batches;


        TorchDataset() {}

        void add_batch(std::vector<at::Tensor> data, std::vector<at::Tensor> target = {});

        void reset();

        // Size of data loaded in memory
        size_t cache_size() const { return _batches.size(); }

        c10::optional<size_t> size() const override {
            return cache_size();
        }

        bool empty() const { return cache_size() == 0; }

        c10::optional<TorchBatch> get_batch(BatchRequestType request) override;

        // Returns a batch containing all the cached data
        TorchBatch get_cached();

        // Split a percentage of this dataset
        TorchDataset split(double start, double stop);
    };



    class TorchInputInterface
    {
    public:
        TorchInputInterface() {}
        TorchInputInterface(const TorchInputInterface &i) {}

        ~TorchInputInterface() {}

        torch::Tensor toLongTensor(std::vector<int64_t> &values) {
            int64_t val_size = values.size();
            return torch::from_blob(&values[0], at::IntList{val_size}, at::kLong);
        }

        TorchDataset _dataset;
        TorchDataset _test_dataset;
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

                std::vector<int64_t> sizes{ _height, _width, 3 };
                at::TensorOptions options(at::ScalarType::Byte);
                at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
                imgt = imgt.toType(at::kFloat).mul(1./255.).permute({2, 0, 1});

                // bgr to rgb
                at::Tensor indexes = torch::ones(3, at::kLong);
                indexes[0] = 2;
                indexes[2] = 0;
                imgt = torch::index_select(imgt, 0, indexes);

                _dataset._batches.push_back(TorchBatch({imgt}, {}));
            }
        }

    public:
    };


    class TxtTorchInputFileConn : public TxtInputFileConn, public TorchInputInterface
    {
    public:
        TxtTorchInputFileConn()
            : TxtInputFileConn() {}
        TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
            : TxtInputFileConn(i), TorchInputInterface(i) 
        {
            _width = i._width;
            _height = i._height;
        }
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

        void transform(const APIData &ad);

        void fill_dataset(TorchDataset &dataset, const std::vector<TxtEntry<double>*> &entries);
    public:
        /** width of the input tensor */
        int _width = 512;
        int _height = 0;
    };
} // namespace dd

#endif // TORCHINPUTCONNS_H
