//
// Created by ramizouari on 14/06/23.
//

#ifndef OPEN_SPIEL_REPRESENTATION_H
#define OPEN_SPIEL_REPRESENTATION_H

#include <string>
#include <map>
#include <unordered_map>
#include <variant>
#include <set>
#include <unordered_set>
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel::interpreter
{

    class RawRepresentation
    {
        const std::string &input;
    public:
        RawRepresentation(const std::string &input);
        const std::string &get() const;
    };


    namespace literals
    {
        RawRepresentation operator""_raw(const char *s, size_t n);
    }


    template<typename K,typename V>
    std::string python_representation(const std::map<K,V> &name);
    template<typename T>
    std::string python_representation(const std::vector<T> &name);
    template<typename T>
    std::string python_representation(const std::set<T> &name);
    template<typename T>
    std::string python_representation(const std::unordered_set<T> &name);
    template<typename K,typename V>
    std::string python_representation(const std::unordered_map<K,V> &name);
    template<typename ...U>
    std::string python_representation(const std::tuple<U...> &name);

    std::string python_representation(const double &name);
    std::string python_representation(const float &name);
    std::string python_representation(const int &name);
    std::string python_representation(const bool &name);
    std::string python_representation(const std::string &name);
    std::string python_representation(const RawRepresentation &name);
    template<typename T>
    std::string python_representation(const std::vector<T> &name)
    {
        std::string s="[";
        for(auto &v:name)
        {
            s+=python_representation(v)+",";
        }
        s+="]";
        return s;
    }

    template<typename K,typename V>
    std::string python_representation(const std::map<K,V> &name)
    {
        std::string s="{";
        for(auto &v:name)
            s+=python_representation(v.first)+":"+python_representation(v.second)+",";
        s+="}";
        return s;
    }

    template<typename K,typename V>
    std::string python_representation(const std::unordered_map<K,V> &name)
    {
        std::string s="{";
        for(auto &v:name)
            s+=python_representation(v.first)+":"+python_representation(v.second)+",";
        s+="}";
        return s;
    }

    template<typename ...U>
    std::string python_representation(const std::tuple<U...> &name)
    {
        std::string s="(";
        std::apply([&s](auto &&...args) {((s+=python_representation(args)+","),...);}, name);
        s+=")";
        return s;
    }

    template<typename ...V>
    std::string python_representation(const std::variant<V...> &name)
    {
        return std::visit([](auto &&arg) {return python_representation(arg);}, name);
    }

    template<typename T>
    std::string python_representation(const std::set<T> &name)
    {
        std::string s="{";
        for(auto &v:name)
            s+=python_representation(v)+",";
        s+="}";
        return s;
    }
    template<typename T>
    std::string python_representation(const std::unordered_set<T> &name)
    {
        std::string s="{";
        for(auto &v:name)
            s+=python_representation(v)+",";
        s+="}";
        return s;
    }


    template<int Rank>
    std::string python_representation_impl(TensorViewConst<Rank> T, size_t dimension)
    {
        if constexpr (Rank == 0)
            return python_representation(T[{}]);
        else
        {
            std::string s="[";
            std::array<int,Rank> shape=T.shape();
            auto subdimension= dimension / shape.front();
            for(int i=0;i<shape.front();i++)
            {
                std::array<int,Rank-1> sub_shape;
                for(int j=0;j<Rank-1;j++)
                    sub_shape[j]=shape[j+1];
                auto data_ptr=std::addressof(T[std::array<int,Rank>{}]);
                absl::Span<const float> sub_span(data_ptr+i*subdimension,subdimension);
                TensorViewConst<Rank-1> T2(sub_span, sub_shape);
                s+=python_representation_impl(T2,subdimension);
                if (i<shape.front()-1)
                    s+=",";
            }
            s+="]";
            return s;
        }
    }

    template<int Rank>
    std::string python_representation_impl(TensorView<Rank> T, size_t dimension)
    {
        return python_representation_impl(TensorViewConst<Rank>(T),dimension);
    }

    template<int Rank>
    std::string python_representation(const TensorView<Rank> &T)
    {
        size_t dimension=1;
        for(auto d:T.shape())
            dimension*=d;
        return python_representation_impl<Rank>(T,dimension);
    }
}

#endif //OPEN_SPIEL_REPRESENTATION_H
