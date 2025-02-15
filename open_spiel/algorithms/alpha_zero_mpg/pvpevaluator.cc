// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pvpevaluator.h"

#include <cstdint>
#include <memory>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/stats.h"

namespace open_spiel::algorithms::mpg
{

    PVPNetEvaluator::PVPNetEvaluator(DeviceManager* device_manager, int batch_size,
                                   int threads, int cache_size, int cache_shards)
        : device_manager_(*device_manager), batch_size_(batch_size),
          queue_(batch_size * threads * 4), batch_size_hist_(batch_size + 1)
    {
      cache_shards = std::max(1, cache_shards);
      cache_.reserve(cache_shards);
      for (int i = 0; i < cache_shards; ++i) {
        cache_.push_back(
            std::make_unique<LRUCache<uint64_t, PVPNetModel::InferenceOutputs>>(
                cache_size / cache_shards));
      }
      if (batch_size_ <= 1) {
        threads = 0;
      }
      inference_threads_.reserve(threads);
      for (int i = 0; i < threads; ++i) {
        inference_threads_.emplace_back([this]() { this->Runner(); });
      }
}

PVPNetEvaluator::~PVPNetEvaluator() {
  stop_.Stop();
  queue_.BlockNewValues();
  queue_.Clear();
  for (auto& t : inference_threads_) {
    t.join();
  }
}

void PVPNetEvaluator::ClearCache() {
  for (auto& c : cache_) {
    c->Clear();
  }
}

LRUCacheInfo PVPNetEvaluator::CacheInfo() {
  LRUCacheInfo info;
  for (auto& c : cache_) {
    info += c->Info();
  }
  return info;
}

std::vector<double> PVPNetEvaluator::Evaluate(const State& state) {
  // TODO(author5): currently assumes zero-sum.
  double p0value = Inference(state).value;
  return {p0value, -p0value};
}

open_spiel::ActionsAndProbs PVPNetEvaluator::Prior(const State& state) {
  return Inference(state).policy;
}

PVPNetModel::InferenceOutputs PVPNetEvaluator::Inference(const State& state)
{
    auto observation=state.ObservationTensor();
    auto state_ = observation.back();
    observation.pop_back();
    PVPNetModel::InferenceInputs inputs = {observation,state_};
    uint64_t key;
    int cache_shard;
    if (!cache_.empty()) {
    key = absl::Hash<PVPNetModel::InferenceInputs>{}(inputs);
    cache_shard = key % cache_.size();
    absl::optional<const PVPNetModel::InferenceOutputs> opt_outputs =
        cache_[cache_shard]->Get(key);
    if (opt_outputs)
        return *opt_outputs;
    }
    PVPNetModel::InferenceOutputs outputs;
    if (batch_size_ <= 1)
        outputs = device_manager_.Get(1)->Inference(std::vector{inputs})[0];
    else
    {
        std::promise<PVPNetModel::InferenceOutputs> prom;
        std::future<PVPNetModel::InferenceOutputs> fut = prom.get_future();
        queue_.Push(QueueItem{inputs, &prom});
        outputs = fut.get();
    }
    if (!cache_.empty())
        cache_[cache_shard]->Set(key, outputs);
    return outputs;
}

void PVPNetEvaluator::Runner() {
  std::vector<PVPNetModel::InferenceInputs> inputs;
  std::vector<std::promise<PVPNetModel::InferenceOutputs>*> promises;
  inputs.reserve(batch_size_);
  promises.reserve(batch_size_);
  while (!stop_.StopRequested()) {
    {
      // Only one thread at a time should be listening to the queue to maximize
      // batch size and minimize latency.
      absl::MutexLock lock(&inference_queue_m_);
      absl::Time deadline = absl::InfiniteFuture();
      for (int i = 0; i < batch_size_; ++i) {
        absl::optional<QueueItem> item = queue_.Pop(deadline);
        if (!item) {  // Hit the deadline.
          break;
        }
        if (inputs.empty()) {
          deadline = absl::Now() + absl::Milliseconds(1);
        }
        inputs.push_back(item->inputs);
        promises.push_back(item->prom);
      }
    }

    if (inputs.empty()) {  // Almost certainly StopRequested.
      continue;
    }

    {
      absl::MutexLock lock(&stats_m_);
      batch_size_stats_.Add(inputs.size());
      batch_size_hist_.Add(inputs.size());
    }

    std::vector<PVPNetModel::InferenceOutputs> outputs =
        device_manager_.Get(inputs.size())->Inference(inputs);
    for (int i = 0; i < promises.size(); ++i) {
      promises[i]->set_value(outputs[i]);
    }
    inputs.clear();
    promises.clear();
  }
}

void PVPNetEvaluator::ResetBatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  batch_size_stats_.Reset();
  batch_size_hist_.Reset();
}

open_spiel::BasicStats PVPNetEvaluator::BatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_stats_;
}

open_spiel::HistogramNumbered PVPNetEvaluator::BatchSizeHistogram() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_hist_;
}

}  // namespace open_spiel
