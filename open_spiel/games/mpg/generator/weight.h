//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_WEIGHT_H
#define OPEN_SPIEL_WEIGHT_H
#include "../mpg.h"
#include "random_pool.h"

namespace open_spiel::mpg
{
    class WeightGenerator: virtual public Seedable
    {
    public:
        virtual ~WeightGenerator()=default;
        virtual WeightType operator()()=0;
    };

    class UniformWeightGenerator:public WeightGenerator, public std::uniform_real_distribution<WeightType>, public DefaultSeedable
    {
        std::mt19937_64 rng;
    public:
        UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed);
        UniformWeightGenerator(WeightType a, WeightType b);
        WeightType operator()() override;

    };

    class NormalWeightGenerator:public WeightGenerator, public std::normal_distribution<WeightType>, public DefaultSeedable
    {
        std::mt19937_64 rng;
    public:
        NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed);
        NormalWeightGenerator(WeightType mean, WeightType std);

        WeightType operator()() override;
    };


    class DiscreteUniformWeightGenerator:public WeightGenerator, public std::uniform_int_distribution<size_t>, public DefaultSeedable
    {
        std::mt19937_64 rng;
    public:
        DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed);
        DiscreteUniformWeightGenerator(size_t a, size_t b);
        WeightType operator()() override;
    };
}

#endif //OPEN_SPIEL_WEIGHT_H
