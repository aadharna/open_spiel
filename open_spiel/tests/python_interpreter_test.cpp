//
// Created by ramizouari on 13/06/23.
//

#include "Python.h"
#include "pybind11/embed.h"
#include "interpreter/interpreter_adapter.h"
#include "reverb/reverb_adapter.h"
#include "open_spiel/utils/tensor_view.h"
namespace py=pybind11;
int main(int argc,char **argv)
{
    using namespace open_spiel::interpreter;
    py::scoped_interpreter guard{};
    InterpreterAdapter A{guard};
    std::cout << "Testing Print" << std::endl;
    A.run("print('Hello World!')");

    std::cout << "Testing Assign and basic operations" << std::endl;
    A.assign("a",3.0);
    A.assign("b",4.0);
    A.run("print(a+b)");


    std::cout << "Testing TensorView" << std::endl;
    std::vector<float> tensor= {1,2,3,4,5,6,7,8};
    absl::Span<float> tensor_span(tensor);
    open_spiel::TensorView<3> tensor_view(tensor_span,{2,2,2},false);
    A.print(tensor_view);
    A.assign("tensor",tensor_view);
    using namespace open_spiel::interpreter::literals;
    A.run("print(tensor)");


    A.run("import reverb");
    A.assign("port",34541);



    std::vector<std::map<std::string,std::string>> data;
    data.push_back({{"a","1"},{"b","2"}});
    data.push_back({{"a","3"},{"b","4"}});
    A.run("client=reverb.Client(f'localhost:{port}')");
    A.assign("data",data);
    A.run(R"(with client.trajectory_writer(num_keep_alive_refs=3) as writer:
  writer.append({'a': 2, 'b': 12})
  writer.append({'a': 3, 'b': 13})
  writer.append({'a': 4,_raw 'b': 14})

  # Create an item referencing all the data.
  writer.create_item(
      table='my_table',
      priority=1.0,
      trajectory={
          'a': writer.history['a'][:],
          'b': writer.history['b'][:],
      })

  # Block until the item has been inserted and confirmed by the server.
  writer.flush())");
}