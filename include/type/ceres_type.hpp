#ifndef LIO_CERES_TYPE_HPP
#define LIO_CERES_TYPE_HPP

#include <ceres/ceres.h>
#include <vector>

namespace lio_data_type
{
  using CeresCostFunctionPtrVector = std::vector<ceres::CostFunction*>;
} // end of namespace lio_data_type 

#endif // end of LIO_CERES_TYPE_HPP
