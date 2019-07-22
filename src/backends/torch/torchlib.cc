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

#include "torchlib.h"

#include <torch/script.h>

#include "outputconnectorstrategy.h"

using namespace torch;

namespace dd
{
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TorchLib(const TorchModel &tmodel)
        : MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TorchModel>(tmodel) 
    {
        this->_libname = "torch";
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TorchLib(TorchLib &&tl) noexcept
        : MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TorchModel>(std::move(tl))
    {
        this->_libname = "torch";
        _traced = std::move(tl._traced);
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TorchLib() 
    {
        
    }

    /*- from mllib -*/
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &ad) 
    {
        bool gpu = false;
        APIData lib_ad = ad.getobj("parameters").getobj("mllib");

        if (lib_ad.has("gpu")) {
            gpu = lib_ad.get("gpu").get<bool>() && torch::cuda::is_available();
        }
        if (lib_ad.has("nclasses")) {
            _nclasses = lib_ad.get("nclasses").get<int>();
        }

        _device = gpu ? torch::Device("cuda") : torch::Device("cpu");

        _traced = torch::jit::load(this->_mlmodel._model_file);
        _traced->to(_device);
        _traced->eval();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) 
    {

    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad, APIData &out) 
    {
        
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::predict(const APIData &ad, APIData &out) 
    {
        APIData params = ad.getobj("parameters");
        APIData output_params = params.getobj("output");

        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy outputc;
        try {
            inputc.transform(ad);
        } catch (...) {
            throw;
        }

        Tensor output = _traced->forward({inputc._in.to(_device)}).toTensor();
        output = torch::softmax(output, 1);
        
        // Output
        std::vector<APIData> results_ads;

        for (int i = 0; i < output.size(0); ++i) {
            std::tuple<Tensor, Tensor> sorted_output = output.slice(0, i, i + 1).sort(1, true);

            APIData results_ad;
            std::vector<double> probs;
            std::vector<std::string> cats;

            if (output_params.has("best")) {
                const int best_count = output_params.get("best").get<int>();

                for (int i = 0; i < best_count; ++i) {
                    probs.push_back(std::get<0>(sorted_output).slice(1, i, i + 1).item<double>());
                    int index = std::get<1>(sorted_output).slice(1, i, i + 1).item<int>();
                    cats.push_back(this->_mlmodel.get_hcorresp(index));
                }
            }

            results_ad.add("uri", inputc._uris.at(results_ads.size()));
            results_ad.add("loss", 0.0);
            results_ad.add("cats", cats);
            results_ad.add("probs", probs);
            results_ad.add("nclasses", 4);

            results_ads.push_back(results_ad);
        }

        outputc.add_results(results_ads);
        outputc.finalize(output_params, out, static_cast<MLModel*>(&this->_mlmodel));

        out.add("status", 0);

        return 0;
    }


    template class TorchLib<ImgTorchInputFileConn,SupervisedOutput,TorchModel>;
}
