//
// Created by ramizouari on 02/07/23.
//

#ifndef OPEN_SPIEL_RANDOM_POOL_H
#define OPEN_SPIEL_RANDOM_POOL_H
#include <random>
#include <mutex>

namespace open_spiel::mpg
{
    struct RandomPool {
        std::mt19937_64 rng;
        std::mutex mutex;
        explicit RandomPool(std::uint64_t seed);
        RandomPool();
        RandomPool(const RandomPool &)=delete;
        RandomPool &operator=(const RandomPool &)=delete;
        std::uint64_t operator()();

    } extern RandomNumberGenerator;

    std::uint64_t NextSeed();


    struct Seedable
    {
        virtual void SetSeed(std::uint64_t seed)=0;
        virtual void SetSeed()=0;
        virtual void SetSeed(const std::string &seed)=0;
    };

    struct DefaultSeedable: virtual public Seedable
    {
        std::mt19937_64 &rng;
        explicit DefaultSeedable(std::mt19937_64 &rng);
        void SetSeed(std::uint64_t seed) override;
        void SetSeed() override;
        void SetSeed(const std::string &seed) override;
    };

    struct RecursiveSeedable: virtual public Seedable
    {
        std::vector<Seedable*> seedables;
        explicit RecursiveSeedable(std::vector<Seedable*> seedables);
        void add(Seedable *seedable);
        void SetSeed(std::uint64_t seed) override;
        void SetSeed() override;
        void SetSeed(const std::string &seed) override;
    };
}

#endif //OPEN_SPIEL_RANDOM_POOL_H
