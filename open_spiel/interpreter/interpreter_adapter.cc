#include "interpreter_adapter.h"

namespace py=pybind11;

namespace open_spiel::interpreter
{
    InterpreterAdapter::InterpreterAdapter(pybind11::scoped_interpreter &guard):guard(guard)
    {
    }

    InterpreterAdapter::~InterpreterAdapter()
    {
    }

    void InterpreterAdapter::run(const std::string &code)
    {
        py::exec(code);
    }


}