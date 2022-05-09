
#include "parquet/column_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "arrow/io/memory.h"
#include "arrow/util/logging.h"
#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/schema.h"
#include "parquet/statistics.h"

#include "arrow/util/optional.h"

namespace parquet {
namespace {

using ::arrow::util::optional;

template <typename DType>
class TypedColumnIndexImpl : public TypedColumnIndex<DType> {
 public:
  using T = typename DType::c_type;

  TypedColumnIndexImpl(const ColumnDescriptor* descr,
                       const BoundaryOrder::type boundary_order,
                       std::vector<bool> null_pages,
                       std::vector<std::string> encoded_min_values,
                       std::vector<std::string> encoded_max_values,
                       std::vector<PageLocation> page_locations,
                       optional<std::vector<int64_t>> null_counts,
                       ::arrow::MemoryPool* pool)
      : descr_(descr),
        boundary_order_(boundary_order),
        null_pages_(std::move(null_pages)),
        encoded_min_values_(std::move(encoded_min_values)),
        encoded_max_values_(std::move(encoded_max_values)),
        page_locations_(std::move(page_locations)),
        null_counts_(std::move(null_counts)),
        pool_(pool) {
    auto page_count = null_pages_.size();
    if ((null_counts_.has_value() && null_counts_->size() != page_count) ||
        encoded_min_values_.size() != page_count ||
        encoded_max_values_.size() != page_count) {
      throw ParquetException("Can't decode column index. Not all sizes are equal.");
    }
  }

  Type::type physical_type() const override { return descr_->physical_type(); }
  const ColumnDescriptor* descr() const override { return descr_; }

  int64_t num_pages() const override { return static_cast<int64_t>(null_pages_.size()); }

  std::shared_ptr<TypedStatistics<DType>> PageStatistics(int64_t page) const override;

 private:
  const ColumnDescriptor* descr_;
  BoundaryOrder::type boundary_order_;
  std::vector<bool> null_pages_;
  std::vector<std::string> encoded_min_values_;
  std::vector<std::string> encoded_max_values_;
  std::vector<PageLocation> page_locations_;
  optional<std::vector<int64_t>> null_counts_;
  ::arrow::MemoryPool* pool_;
};

template <typename DType>
std::shared_ptr<TypedStatistics<DType>> TypedColumnIndexImpl<DType>::PageStatistics(
    int64_t page) const {
  bool is_null_page = null_pages_[page];
  bool has_null_count = null_counts_.has_value();
  int64_t num_values = page_locations_[page].num_rows, null_count = 0;

  if (is_null_page) {
    std::swap(null_count, num_values);
    has_null_count = true;
  } else if (has_null_count) {
    null_count = (*null_counts_)[page];
    num_values -= null_count;
  }

  return MakeStatistics<DType>(descr_, encoded_min_values_[page],
                               encoded_max_values_[page], num_values, null_count, 0,
                               /*has_min_max*/ !is_null_page, has_null_count,
                               /*has_distinct_count*/ false, pool_);
}

}  // namespace

std::shared_ptr<ColumnIndex> ColumnIndex::Make(
    const ColumnDescriptor* descr, const BoundaryOrder::type boundary_order,
    std::vector<bool> null_pages, std::vector<std::string> encoded_min_values,
    std::vector<std::string> encoded_max_values, std::vector<PageLocation> page_locations,
    optional<std::vector<int64_t>> null_counts, ::arrow::MemoryPool* pool) {
#define MAKE_COL_INDEX(CAP_TYPE, KLASS)                                            \
  case Type::CAP_TYPE:                                                             \
    return std::make_shared<TypedColumnIndexImpl<KLASS>>(                          \
        descr, boundary_order, null_pages, encoded_min_values, encoded_max_values, \
        page_locations, null_counts, pool)

  switch (descr->physical_type()) {
    MAKE_COL_INDEX(BOOLEAN, BooleanType);
    MAKE_COL_INDEX(INT32, Int32Type);
    MAKE_COL_INDEX(INT64, Int64Type);
    MAKE_COL_INDEX(INT96, Int96Type);
    MAKE_COL_INDEX(FLOAT, FloatType);
    MAKE_COL_INDEX(DOUBLE, DoubleType);
    MAKE_COL_INDEX(BYTE_ARRAY, ByteArrayType);
    MAKE_COL_INDEX(FIXED_LEN_BYTE_ARRAY, FLBAType);
    default:
      break;
  }
#undef MAKE_COL_INDEX
  DCHECK(false) << "Cannot reach here";
  return nullptr;
}

}  // namespace parquet
