#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <exception>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)
#endif

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define CAFFE1_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace caffe {

using std::string;
using std::vector;

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
