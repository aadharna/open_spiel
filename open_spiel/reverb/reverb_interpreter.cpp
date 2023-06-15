//
// Created by ramizouari on 13/06/23.
//

#include "reverb_interpreter.h"

namespace open_spiel {
    namespace interpreter
    {


        ReverbInterpreter::ReverbInterpreter(const std::string&host,const std::string &port,pybind11::scoped_interpreter &guard):InterpreterAdapter(guard),
            host(host),port(port)
        {
            run("import reverb");
            assign("server_host",host+":"+port);
            //Client
            run("client=reverb.Client(server_host)");
        }
    } // open_spiel
} // interpreter