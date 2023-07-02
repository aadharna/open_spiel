//
// Created by ramizouari on 23/06/23.
//

#include "weight.h"
#include "random_pool.h"

namespace open_spiel::mpg
{
    DiscreteUniformWeightGenerator::DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed): rng(seed),std::uniform_int_distribution<size_t>(a,b),
                                                                                                            DefaultSeedable(rng)
    {
    }


    WeightType DiscreteUniformWeightGenerator::operator()() {
        return std::uniform_int_distribution<size_t>::operator()(rng);
    }

    DiscreteUniformWeightGenerator::DiscreteUniformWeightGenerator(size_t a, size_t b) : DiscreteUniformWeightGenerator(a,b,NextSeed()) {

    }

    WeightType NormalWeightGenerator::operator()() {
        return std::normal_distribution<WeightType>::operator()(rng);
    }

    NormalWeightGenerator::NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed) : rng(seed),std::normal_distribution<WeightType>(mean,std),
                                                                                                        DefaultSeedable(rng)
    {
    }

    NormalWeightGenerator::NormalWeightGenerator(WeightType mean, WeightType std) : NormalWeightGenerator(mean,std,NextSeed()) {

    }

    UniformWeightGenerator::UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed): rng(seed),std::uniform_real_distribution<WeightType>(a,b),
                                                                                                    DefaultSeedable(rng)
    {
    }

    WeightType UniformWeightGenerator::operator()()
    {
        return std::uniform_real_distribution<WeightType>::operator()(rng);
    }

    UniformWeightGenerator::UniformWeightGenerator(WeightType a, WeightType b) : UniformWeightGenerator(a,b,NextSeed()) {

    }

}