//
// Created by ramizouari on 14/06/23.
//

#include "representation.h"


namespace open_spiel::interpreter
{
    std::string python_representation(const float &name)
    {
        return std::to_string(name);
    }

    std::string python_representation(const double &name)
    {
        return std::to_string(name);
    }

    std::string python_representation(const int &name)
    {
        return std::to_string(name);
    }

    std::string python_representation(const bool &name)
    {
        return name?"True":"False";
    }

    std::string python_representation(const std::string &name)
    {
        return "'"+name+"'";
    }

    std::string python_representation(const RawRepresentation &name)
    {
        return name.get();
    }

    const std::string &RawRepresentation::get() const
    {
        return input;
    }

    RawRepresentation::RawRepresentation(const std::string &input):input(input)
    {

    }

    RawRepresentation literals::operator""_raw(const char *s, size_t n)
    {
        return RawRepresentation(std::string(s,n));
    }
}