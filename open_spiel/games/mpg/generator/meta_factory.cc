//
// Created by ramizouari on 23/06/23.
//
#include "meta_factory.h"
#include "mpg/mpg_generator.h"
#include "environment.h"
#include "mpg/generator/meta_factory.h"


namespace open_spiel::mpg
{
    void ParserMetaFactory::RegisterFactory(const std::string &name, std::shared_ptr<ParametricMetaFactory> factory)
    {
        factories.emplace(name,std::move(factory));
    }

    ParserMetaFactory::ParserMetaFactory()
    {
        class ExampleMetaFactory : public ParametricMetaFactory
        {
        public:
            std::shared_ptr<const Game> GameFromArgs(const std::vector<double> &args, const GameParameters &params) override
            {
                return std::make_shared<MPGMetaGame>(params, std::make_unique<ExampleEnvironmentFactory>());
            }

            size_t MinArgs() const override
            {
                return 0;
            }

            size_t MaxArgs() const override
            {
                return 0;
            }
        };
        RegisterFactory("gnp",std::make_shared<UniformGnpMetaFactory>());
        RegisterFactory("usgnp",std::make_shared<UniformlyStochasticUniformGnpMetaFactory>());
        RegisterFactory("example",std::make_shared<ExampleMetaFactory>());
        RegisterFactory("file",std::make_shared<UnimplementedMetaFactory>());
        RegisterFactory("dataset",std::make_shared<DatasetMetaFactory>());
        RegisterFactory("gnc",std::make_shared<UniformGncMetaFactory>());
        RegisterFactory("usgnc",std::make_shared<UniformlyStochasticUniformGncMetaFactory>());
    }

    std::shared_ptr<const Game> ParserMetaFactory::CreateGame(const GameParameters &params)
    {
        if(!params.count("generator"))
            throw std::invalid_argument("generator parameter is required");

        if(!factories.count(params.at("generator").string_value()) && !string_factories.count(params.at("generator").string_value()))
            throw std::invalid_argument(absl::StrCat(params.at("generator").string_value()," is not a registered generator"));
        if(factories.count(params.at("generator").string_value()))
        {
            std::string game_generator=params.at("generator").string_value();
            auto factory=factories.at(game_generator);
            if(factory->MinArgs()==0 && !params.count("generator_params"))
                return factory->CreateGame(params);
            else
            {
                std::string generator_params=params.at("generator_params").string_value();
                std::stringstream ss(generator_params);
                std::vector<double> args(factory->MinArgs());
                args.reserve(factory->MaxArgs());
                try
                {
                    ss.exceptions(std::ios::failbit);
                    for(int i=0;i<factory->MinArgs();i++)
                        ss >> args[i];
                    for(int i=factory->MinArgs();i<factory->MaxArgs() && !ss.eof();i++)
                    {
                        int r;
                        ss >> r;
                        args.push_back(r);
                    }
                    return factory->GameFromArgs(args,params);
                }
                catch (std::ios::failure& e)
                {
                    throw std::invalid_argument("Invalid generator_params for gnp: "+generator_params);
                }
            }
        }
        else
        {
            std::string game_generator=params.at("generator").string_value();
            auto factory=string_factories.at(game_generator);
            if(factory->MinArgs()==0 && !params.count("generator_params"))
                return factory->CreateGame(params);
            else
            {
                std::string generator_params=params.at("generator_params").string_value();
                std::stringstream ss(generator_params);
                std::vector<std::string> args(factory->MinArgs());
                args.reserve(factory->MaxArgs());
                try
                {
                    ss.exceptions(std::ios::failbit);
                    for(int i=0;i<factory->MinArgs();i++)
                        ss >> args[i];
                    for(int i=factory->MinArgs();i<factory->MaxArgs() && !ss.eof();i++)
                    {
                        std::string r;
                        ss >> r;
                        args.push_back(r);
                    }
                    return factory->GameFromArgs(args,params);
                }
                catch (std::ios::failure& e)
                {
                    throw std::invalid_argument("Invalid generator_params for gnp: "+generator_params);
                }
            }
        }
    }

    void
    ParserMetaFactory::RegisterFactory(const std::string &name, std::shared_ptr<StringParametricMetaFactory> factory) {
        string_factories.emplace(name,std::move(factory));
    }

    std::shared_ptr<const Game> UnimplementedMetaFactory::GameFromArgs(const std::vector<double> &args, const GameParameters &params)
    {
        throw std::invalid_argument(absl::StrCat(params.at("generator").string_value()," is registered but not implemented"));
    }

    std::size_t UnimplementedMetaFactory::MinArgs() const {
        return 0;
    }

    std::size_t UnimplementedMetaFactory::MaxArgs() const {
        return 0;
    }


    std::size_t UniformGnpMetaFactory::MinArgs() const {
        return 4;
    }

    std::size_t UniformGnpMetaFactory::MaxArgs() const {
        return 5;
    }


    std::size_t UniformlyStochasticUniformGnpMetaFactory::MinArgs() const {
        return 6;
    }

    std::size_t UniformlyStochasticUniformGnpMetaFactory::MaxArgs() const {
        return 7;
    }

    std::shared_ptr<const Game> ParametricMetaFactory::CreateGame(const GameParameters &params)
    {
        parserMetaFactory->RegisterFactory(params.at("generator").string_value(),shared_from_this());
        return parserMetaFactory->CreateGame(params);
    }


    std::shared_ptr<const Game> StringParametricMetaFactory::CreateGame(const GameParameters &params)
    {
        parserMetaFactory->RegisterFactory(params.at("generator").string_value(),shared_from_this());
        return parserMetaFactory->CreateGame(params);
    }
}