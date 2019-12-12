#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

namespace py = boost::python;

// Converts a C++ vector to a python list
template <class T>
inline
py::list toPythonList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    py::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

template<class T >
inline
std::vector< T > to_std_vector( const py::object& iterable )
{
    return std::vector< T >( py::stl_input_iterator< T >( iterable ),
                             py::stl_input_iterator< T >( ) );
}
