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
    // ======= TORCH MODULE


    void add_parameters(std::shared_ptr<torch::jit::script::Module> module, std::vector<Tensor> &params) { 
        for (const auto &slot : module->get_parameters()) {
            params.push_back(slot.value().toTensor());
        }
        for (auto child : module->get_modules()) {
            add_parameters(child, params);
        }
    }

    c10::IValue TorchModule::forward(std::vector<c10::IValue> source) 
    {
        if (_traced)
        {
            source = { _traced->forward(source) };
        }
        c10::IValue out_val = source.at(0);
        if (_classif)
        {
            if (!out_val.isTensor())
                throw MLLibInternalException("Model returned an invalid output. Please check your model.");
            out_val = _classif->forward(out_val.toTensor());
        }
        return out_val;
    }

    std::vector<Tensor> TorchModule::parameters() 
    {
        std::vector<Tensor> params;
        if (_traced)
            add_parameters(_traced, params);
        if (_classif)
        {
            auto classif_params = _classif->parameters();
            params.insert(params.end(), classif_params.begin(), classif_params.end());
        }
        return params;
    }

    void TorchModule::save(const std::string &filename) 
    {
       // Not yet implemented 
    }

    void TorchModule::load(const std::string &filename) 
    {
        // Not yet implemented
    }


    // ======= TORCHLIB

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
        _module = std::move(tl._module);
        _nclasses = tl._nclasses;
        _device = tl._device;
        _attention = tl._attention;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TorchLib() 
    {
        
    }

    /*- from mllib -*/
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &lib_ad) 
    {
        bool gpu = false;

        if (lib_ad.has("gpu")) {
            gpu = lib_ad.get("gpu").get<bool>() && torch::cuda::is_available();
        }
        if (lib_ad.has("nclasses")) {
            _nclasses = lib_ad.get("nclasses").get<int>();
        }

        _device = gpu ? torch::Device("cuda") : torch::Device("cpu");

        if (typeid(TInputConnectorStrategy) == typeid(TxtTorchInputFileConn)) {
            _attention = true;
        }

        _module._traced = torch::jit::load(this->_mlmodel._model_file, _device);
        _module._traced->eval();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) 
    {

    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad, APIData &out) 
    {
        TInputConnectorStrategy inputc(this->_inputc);
        try
        {
            inputc.transform(ad);
        }
        catch (...)
        {
            throw;
        }

        APIData ad_mllib = ad.getobj("parameters").getobj("mllib");

        // solver params
        int64_t iterations = 100;
        std::string solver_type = "SGD";
        int64_t base_lr = 0.0001;

        if (ad_mllib.has("solver"))
        {
            APIData ad_solver = ad_mllib.getobj("solver");
            if (ad_solver.has("iterations"))
                iterations = ad_solver.get("iterations").get<int>();
            if (ad_solver.has("solver_type"))
                solver_type = ad_solver.get("solver_type").get<std::string>();
            if (ad_solver.has("iterations"))
                base_lr = ad_solver.get("base_lr").get<double>();
        }

        // create solver
        // no care about solver type yet
        optim::Adam optimizer(_module.parameters(), optim::AdamOptions(base_lr));

        /* for (int64_t epoch = 0; epoch < iterations; ++epoch) {
            int64_t batch_index = 0;
            double loss_sum = 0;

            for (BertTensorExample &example : *train_data) {
                Tensor x = example.data["input_ids"].to(device);
                Tensor att = example.data["attention"].to(device);
                Tensor y = example.target.to(device);

                Tensor y_pred = model(x, att);

                auto loss = torch::mse_loss(y_pred, y);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                double loss_value = loss.item<double>();
                loss_sum += loss_value;

                if (batch_index % 20 == 0) {
                    std::printf("E: %ld/%ld, B: %ld, loss = %f\n", epoch, EPOCH_COUNT, batch_index, loss.item<double>());
                }
                batch_index++;
            }

            std::cout << "Saving checkpoint. Loss=" << loss_sum / batch_index << std::endl;
            model->save(MODEL_PATH);
        }*/

        return 0;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::predict(const APIData &ad, APIData &out) 
    {
        APIData params = ad.getobj("parameters");
        APIData output_params = params.getobj("output");

        if (output_params.has("measure"))
        {
            test(ad, out);
            return 0;
        } 

        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy outputc;
        try {
            inputc.transform(ad);
        } catch (...) {
            throw;
        }

        // TODO less boilerplate
        // FIXME an assert can fail while accessing optional (dede crash)
        inputc._dataset.reset();
        auto data = inputc._dataset.get_batch(std::vector<size_t>{*inputc._dataset.size()})->data;
        std::vector<c10::IValue> in_vals;
        for (Tensor tensor : data)
            in_vals.push_back(tensor.to(_device));
        Tensor output = torch::softmax(_module.forward(in_vals).toTensor(), 1);
        
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

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::test(const APIData &ad, APIData &out) 
    {
        /* APIData ad_res;
        APIData ad_out = ad.getobj("params").getobj("output");

        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy outputc;
        try {
            inputc.transform(ad);
        } catch (...) {
            throw;
        }

        at::Tensor &labels = inputc._labels;
        int test_size = labels.size(0);

        int batch_size = 5;
        int batch_count = (test_size - 1 / batch_size + 1;

        double valid = 0;

        for (int i = 0; i < batch_count; ++i) {
            int slice_start = i * batch_size;
            int slice_end = (i+1) * batch_size;

            if (slice_end > test_size) {
                slice_end = test_size;
            }

            std::vector<c10::IValue> in_vals;
            in_vals.push_back(inputc._in.to(_device).slice(0, slice_start, slice_end));

            if (_attention) {
                // token_type_ids
                in_vals.push_back(torch::zeros_like(inputc._in, at::kLong).to(_device)
                    .slice(0, slice_start, slice_end));
                in_vals.push_back(inputc._attention_mask.to(_device)
                    .slice(0, slice_start, slice_end));
            }

            Tensor output = run_classification_model(in_vals);
            Tensor max_ids = output.argmax(1).to(kLong);
            Tensor labels = inputc._labels.slice(slice_start, slice_end).to(kLong);

            for (int j = 0; j < max_ids.size(0); ++j) {
                if (labels[j].item<int64_t>() == max_ids[j].item<int64_t>()) {
                    ++valid;
                }
            }
        }

        double accuracy = valid / test_size * 100;
        ad_res.add("acc", accuracy);
        SupervisedOutput::measure(ad_res, ad_out, out);*/
        return 0;
    }

    template class TorchLib<ImgTorchInputFileConn,SupervisedOutput,TorchModel>;
    template class TorchLib<TxtTorchInputFileConn,SupervisedOutput,TorchModel>;
}
