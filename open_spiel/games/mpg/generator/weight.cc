//
// Created by ramizouari on 23/06/23.
//

#include "weight.h"


namespace open_spiel::mpg
{
    DiscreteUniformWeightGenerator::DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed): rng(seed),std::uniform_int_distribution<size_t>(a,b)
    {
    }


    WeightType DiscreteUniformWeightGenerator::operator()() {
        return std::uniform_int_distribution<size_t>::operator()(rng);
    }

    WeightType NormalWeightGenerator::operator()() {
        return std::normal_distribution<WeightType>::operator()(rng);
    }

    NormalWeightGenerator::NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed) : rng(seed),std::normal_distribution<WeightType>(mean,std)
    {
    }

    UniformWeightGenerator::UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed): rng(seed),std::uniform_real_distribution<WeightType>(a,b)
    {
    }

    WeightType UniformWeightGenerator::operator()()
    {
        return std::uniform_real_distribution<WeightType>::operator()(rng);
    }

}