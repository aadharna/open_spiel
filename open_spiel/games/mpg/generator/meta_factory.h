//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_META_FACTORY_H
#define OPEN_SPIEL_META_FACTORY_H
#include "../mpg.h"


namespace open_spiel::mpg
{
    class MetaFactory
    {
    public:
        virtual ~MetaFactory() = default;
        virtual std::shared_ptr<const Game> CreateGame(const GameParameters& params)  = 0;

    };


    class ParametricMetaFactory : public MetaFactory, std::enable_shared_from_this<ParametricMetaFactory>
    {
    public:
        virtual std::shared_ptr<const Game> GameFromArgs(const std::vector<double> &args,const GameParameters &params) = 0;
        std::shared_ptr<const Game> CreateGame(const GameParameters& params) override;
        virtual ~ParametricMetaFactory() = default;
        virtual size_t MinArgs() const = 0;
        virtual size_t MaxArgs() const = 0;
    };

    class StringParametricMetaFactory: MetaFactory, std::enable_shared_from_this<StringParametricMetaFactory>
            {
                    public:
                    virtual std::shared_ptr<const Game> GameFromArgs(const std::vector<std::string> &args,const GameParameters &params) = 0;
                    std::shared_ptr<const Game> CreateGame(const GameParameters& params) override;
                    virtual ~StringParametricMetaFactory() = default;
                    virtual size_t MinArgs() const = 0;
                    virtual size_t MaxArgs() const = 0;
            };

    class ParserMetaFactory : public MetaFactory
    {

    public:
        ParserMetaFactory();
        std::shared_ptr<const Game> CreateGame(const GameParameters& params) override;
        void RegisterFactory(const std::string &name, std::shared_ptr<ParametricMetaFactory> factory);
        void RegisterFactory(const std::string &name, std::shared_ptr<StringParametricMetaFactory> factory);
    private:
        std::map<std::string,std::shared_ptr<ParametricMetaFactory>> factories;
        std::map<std::string,std::shared_ptr<StringParametricMetaFactory>> string_factories;
    };

    class UnimplementedMetaFactory : public ParametricMetaFactory
    {
    public:
        std::shared_ptr<const Game> GameFromArgs(const std::vector<double> &args,const GameParameters &params) override;
        std::size_t MinArgs() const override;
        std::size_t MaxArgs() const override;

    };

}

#endif //OPEN_SPIEL_META_FACTORY_H
