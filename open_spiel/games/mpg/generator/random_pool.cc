//
// Created by ramizouari on 02/07/23.
//

#include "random_pool.h"


namespace open_spiel::mpg
{
    std::uint64_t RandomPool::operator()()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return rng();
    }

    RandomPool::RandomPool(std::uint64_t seed):rng(seed)
    {

    }

    RandomPool::RandomPool() :rng(std::random_device()()){
    }
    RandomPool RandomNumberGenerator;

    std::uint64_t NextSeed()
    {
        return RandomNumberGenerator();
    }

    DefaultSeedable::DefaultSeedable(std::mt19937_64 &rng):rng(rng) {

    }

    void DefaultSeedable::SetSeed(std::uint64_t seed) {
        rng.seed(seed);
    }

    void DefaultSeedable::SetSeed() {
        SetSeed(NextSeed());
    }

    void DefaultSeedable::SetSeed(const std::string &seed) {
        rng.seed(std::hash<std::string>{}(seed));
    }

    RecursiveSeedable::RecursiveSeedable(std::vector<Seedable *> seedables):seedables(std::move(seedables)) {

    }

    void RecursiveSeedable::add(Seedable *seedable) {
        seedables.push_back(seedable);
    }

    void RecursiveSeedable::SetSeed(std::uint64_t seed) {
        for(auto &s:seedables)
            s->SetSeed(seed);
    }

    void RecursiveSeedable::SetSeed() {
        for(auto &s:seedables)
            s->SetSeed();
    }

    void RecursiveSeedable::SetSeed(const std::string &seed) {
        for(auto &s:seedables)
            s->SetSeed(seed);
    }
}