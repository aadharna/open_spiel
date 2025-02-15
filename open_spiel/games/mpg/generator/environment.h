//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_GENERATOR_ENVIRONMENT_H
#define OPEN_SPIEL_GENERATOR_ENVIRONMENT_H
#include "../mpg.h"
#include "random_pool.h"


namespace open_spiel::mpg
{
    class EnvironmentFactory : public Seedable
    {
    public:
        virtual ~EnvironmentFactory()=default;
        virtual std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)=0;
        void SetSeed(std::uint64_t seed) override;
        void SetSeed() override;
        void SetSeed(const std::string &seed) override;
    };


    class GeneratorEnvironmentFactory : public EnvironmentFactory
    {
        std::shared_ptr<class WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        explicit GeneratorEnvironmentFactory(std::shared_ptr<class WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed);
        explicit GeneratorEnvironmentFactory(std::shared_ptr<class WeightedGraphGenerator> weighted_graph_generator);

        [[nodiscard]] std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)  override;
        void SetSeed(std::uint64_t seed) override;
        void SetSeed() override;
        void SetSeed(const std::string &seed) override;
    };

    class UniformGnpEnvironmentFactory : public GeneratorEnvironmentFactory
    {

    public:
        UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b, std::uint64_t seed);
        UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b);
    };

    class UniformGncEnvironmentFactory : public GeneratorEnvironmentFactory
    {

    public:
        UniformGncEnvironmentFactory(NodeType n, WeightType c, WeightType a, WeightType b, std::uint64_t seed);
        UniformGncEnvironmentFactory(NodeType n, WeightType c, WeightType a, WeightType b);

    };

    class UniformlyStochasticUniformGnpEnvironmentFactory : public GeneratorEnvironmentFactory
    {
    public:
        UniformlyStochasticUniformGnpEnvironmentFactory(NodeType n_min, NodeType n_max, double p_min, double p_max,
                                                        WeightType a, WeightType b, std::uint64_t seed);
        UniformlyStochasticUniformGnpEnvironmentFactory(NodeType n_min, NodeType n_max, double p_min, double p_max,
                                                        WeightType a, WeightType b);
    };

    class UniformlyStochasticUniformGncEnvironmentFactory : public GeneratorEnvironmentFactory
    {
    public:
        UniformlyStochasticUniformGncEnvironmentFactory(NodeType n_min, NodeType n_max, double c_min, double c_max,
                                                        WeightType a, WeightType b, std::uint64_t seed);
        UniformlyStochasticUniformGncEnvironmentFactory(NodeType n_min, NodeType n_max, double c_min, double c_max,
                                                        WeightType a, WeightType b);
    };

    using USUGEnvironmentFactory = UniformlyStochasticUniformGnpEnvironmentFactory;
    using UGEnvironmentFactory = UniformGnpEnvironmentFactory;

    using USparseEnvironmentFactory = UniformGncEnvironmentFactory;
    using UDenseEnvironmentFactory = UniformGnpEnvironmentFactory;
    using USUSparseEnvironmentFactory = UniformlyStochasticUniformGncEnvironmentFactory;
    using USUDenseEnvironmentFactory = UniformlyStochasticUniformGnpEnvironmentFactory;

    class DeterministicEnvironmentFactory : public EnvironmentFactory
    {
    public:
        ~DeterministicEnvironmentFactory() override = default;
        std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame& metaGame)  override = 0;
    };

    class ExampleEnvironmentFactory : public DeterministicEnvironmentFactory
    {
    public:
        ~ExampleEnvironmentFactory() override = default;
        std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame& metaGame) override;
    };


    class DatasetEnvironmentFactory : public DeterministicEnvironmentFactory
    {
        std::string dataset_path;
        std::vector<std::string> dataset_files;
        std::mt19937_64 rng;
    public:
        explicit DatasetEnvironmentFactory(std::string dataset_path, std::uint64_t seed=0);
        std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame& metaGame) override;
    };
}

#endif //OPEN_SPIEL_GENERATOR_ENVIRONMENT_H
