//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_WEIGHT_H
#define OPEN_SPIEL_WEIGHT_H
#include "../mpg.h"

namespace open_spiel::mpg
{
    class WeightGenerator
    {
    public:
        virtual ~WeightGenerator()=default;
        virtual WeightType operator()()=0;
    };

    class UniformWeightGenerator:public WeightGenerator, public std::uniform_real_distribution<WeightType>
    {
        std::mt19937_64 rng;
    public:
        UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed = 0);
        WeightType operator()() override;
    };

    class NormalWeightGenerator:public WeightGenerator, public std::normal_distribution<WeightType>
    {
        std::mt19937_64 rng;
    public:
        NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed = 0);
        WeightType operator()() override;
    };


    class DiscreteUniformWeightGenerator:public WeightGenerator, public std::uniform_int_distribution<size_t>
    {
        std::mt19937_64 rng;
    public:
        DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed = 0);
        WeightType operator()() override;
    };
}

#endif //OPEN_SPIEL_WEIGHT_H
