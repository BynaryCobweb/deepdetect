/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef NATIVE_NET_H
#define NATIVE_NET_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "../torchinputconns.h"

using namespace torch;
using namespace torch::nn;
using namespace c10;

namespace dd
{

  class NativeModule : public virtual torch::nn::Module
  {
  public:
    /**
     * \brief forward pass over the
     * @param input tensor
     * @return value of output
     */
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /**
     * \brief extract layer from net
     * @param input
     * @param name of data to extract
     * @return extracted tensor
     */
    virtual torch::Tensor extract(torch::Tensor x, std::string extract_layer)
        = 0;

    /**
     * \brief check is string correspond to some layer in the net
     * @param the name of the data node
     * @return true if it exists in the net
     */
    virtual bool extractable(std::string extract_layer) const = 0;

    /**
     * \brief return all candidates for extraction, ie all data nodes of the
     * net
     */
    virtual std::vector<std::string> extractable_layers() const = 0;

    virtual ~NativeModule() = default;

    virtual torch::Tensor cleanup_output(torch::Tensor output) = 0;

    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
        = 0;

    virtual void update_input_connector(TorchInputInterface &inputc) = 0;
  };

  template <typename Derived>
  class DDCloneable : public virtual torch::nn::Module
  {
  public:
    virtual ~DDCloneable() = default;
    using Module::Module;

    /// `reset()` must perform initialization of all members with reference
    /// semantics, most importantly parameters, buffers and submodules.
    virtual void reset() = 0;

    /// Performs a recursive "deep copy" of the `Module`, such that all
    /// parameters and submodules in the cloned module are different from those
    /// in the original module.
    std::shared_ptr<Module> clone(const optional<Device> &device
                                  = nullopt) const override
    {
      NoGradGuard no_grad;

      const auto &self = static_cast<const Derived &>(*this);
      auto copy = std::make_shared<Derived>(self);
      copy->parameters_.clear();
      copy->buffers_.clear();
      copy->children_.clear();
      copy->reset();
      // [[this pointer note]]
      // Don't remove 'this' pointer, nvcc needs it to be explicitly given in
      // some envs. eg. ubuntu 16.04 + gcc 5.x + cuda 9.2
      //     ubuntu 16.04 + gcc 7.x + cuda 9.2
      TORCH_CHECK(
          copy->parameters_.size() == this->parameters_.size(),
          "The cloned module does not have the same number of "
          "parameters as the original module after calling reset(). "
          "Are you sure you called register_parameter() inside reset() "
          "and not the constructor?");
      for (const auto &parameter : named_parameters(/*recurse=*/false))
        {
          auto &tensor = *parameter;
          auto data = device && tensor.device() != *device
                          ? tensor.to(*device)
                          : autograd::Variable(tensor).clone();
          copy->parameters_[parameter.key()].set_data(data);
        }
      // Don't remove 'this' pointer. See [[this pointer note]]
      TORCH_CHECK(copy->buffers_.size() == this->buffers_.size(),
                  "The cloned module does not have the same number of "
                  "buffers as the original module after calling reset(). "
                  "Are you sure you called register_buffer() inside reset() "
                  "and not the constructor?");
      for (const auto &buffer : named_buffers(/*recurse=*/false))
        {
          auto &tensor = *buffer;
          auto data = device && tensor.device() != *device
                          ? tensor.to(*device)
                          : autograd::Variable(tensor).clone();
          copy->buffers_[buffer.key()].set_data(data);
        }
      // Don't remove 'this' pointer. See [[this pointer note]]
      TORCH_CHECK(
          copy->children_.size() == this->children_.size(),
          "The cloned module does not have the same number of "
          "child modules as the original module after calling reset(). "
          "Are you sure you called register_module() inside reset() "
          "and not the constructor?");
      // Don't remove 'this' pointer. See [[this pointer note]]
      for (const auto &child : this->children_)
        {
          copy->children_[child.key()]->clone_(*child.value(), device);
        }
      return copy;
    }

  private:
    void clone_(Module &other, const optional<Device> &device) final
    {
      // Here we are *pretty* certain that `other's` type is `Derived` (because
      // it was registered under the same name as `this`), but you never know
      // what crazy things `reset()` does, so `dynamic_cast` just to be safe.
      auto clone = std::dynamic_pointer_cast<Derived>(other.clone(device));
      TORCH_CHECK(
          clone != nullptr,
          "Attempted to clone submodule, but it is of a "
          "different type than the submodule it was to be cloned into");
      static_cast<Derived &>(*this) = std::move(*clone);
    }
  };

  template <typename T>
  class NativeModuleImpl : public NativeModule, public DDCloneable<T>
  {
  };
}

#endif
