#ifndef OPEN_SPIEL_INTERPRETER_INTERPRETER_ADAPTER_H_
#define OPEN_SPIEL_INTERPRETER_INTERPRETER_ADAPTER_H_
#include "pybind11/embed.h"
#include "representation.h"

namespace open_spiel::interpreter
{



    class InterpreterAdapter
    {
        public:
            InterpreterAdapter(pybind11::scoped_interpreter &guard);
            ~InterpreterAdapter();
            void run(const std::string &code);

            template<typename T>
            void assign(const std::string &name, const T &value)
            {
                pybind11::exec(name+"="+python_representation(value));
            }

            template<typename T>
            void print(const T &value)
            {
                pybind11::print(python_representation(value));
            }

            void import(const std::string &name)
            {
                pybind11::exec("import "+name);
            }

            void from_import(const std::string &name, const std::string &module)
            {
                pybind11::exec("from "+module+" import "+name);
            }

        private:
            pybind11::scoped_interpreter &guard;
    };
}
#endif //OPEN_SPIEL_INTERPRETER_INTERPRETER_ADAPTER_H_