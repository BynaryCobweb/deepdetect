#include "db.hpp"
#include "db_lmdb.hpp"

#include <string>

#define USE_LMDB

namespace caffe { namespace db {

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (backend == "lmdb") {
    return new LMDB();
  }
#endif  // USE_LMDB
  // #### LOG(ERROR) << "Unknown database backend";
  // #### LOG(FATAL) << "fatal error";
  return NULL;
}

}  // namespace db
}  // namespace caffe
