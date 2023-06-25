//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_GENERATOR_ENVIRONMENT_H
#define OPEN_SPIEL_GENERATOR_ENVIRONMENT_H
#include "../mpg.h"


namespace open_spiel::mpg
{
    class EnvironmentFactory
    {
    public:
        virtual ~EnvironmentFactory()=default;
        virtual std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)=0;
    };


    class GeneratorEnvironmentFactory : public EnvironmentFactory
    {
        std::shared_ptr<class WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        explicit GeneratorEnvironmentFactory(std::shared_ptr<class WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed=0);
        [[nodiscard]] std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)  override;
    };

    class UniformGnpEnvironmentFactory : public GeneratorEnvironmentFactory
    {

    public:
        UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b, std::uint64_t seed = 0);

    };

    class UniformlyStochasticUniformGnpEnvironmentFactory : public GeneratorEnvironmentFactory
    {
    public:
        UniformlyStochasticUniformGnpEnvironmentFactory(NodeType n_min, NodeType n_max, double p_min, double p_max,
                                                        WeightType a, WeightType b, std::uint64_t seed = 0);
    };

    using USUGEnvironmentFactory = UniformlyStochasticUniformGnpEnvironmentFactory;
    using UGEnvironmentFactory = UniformGnpEnvironmentFactory;

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
