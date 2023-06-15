//
// Created by ramizouari on 13/06/23.
//

#ifndef OPEN_SPIEL_REVERB_INTERPRETER_H
#define OPEN_SPIEL_REVERB_INTERPRETER_H
#include "../interpreter/interpreter_adapter.h"
#include "utils/tensor_view.h"

namespace open_spiel {
    namespace interpreter {


        class ReverbInterpreter:public InterpreterAdapter {
        public:
            ReverbInterpreter(const std::string &host, const std::string &port,pybind11::scoped_interpreter &guard);
            template<typename A,typename B>
            void insert(const std::string &table_name,const A &keys,const B &values)
            {
                assign("keys", python_representation(keys));
                assign("values", python_representation(values));
                run("client.insert(table_name, keys, values)");
            }

        private:
            std::string host;
            std::string port;
        };

    } // interpreter

    namespace reverb
    {
        using open_spiel::interpreter::ReverbInterpreter;
    }
} // open_spiel

#endif //OPEN_SPIEL_REVERB_INTERPRETER_H
