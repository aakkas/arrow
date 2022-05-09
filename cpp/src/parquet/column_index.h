#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "arrow/util/optional.h"
#include "parquet/platform.h"
#include "parquet/statistics.h"
#include "parquet/types.h"

namespace parquet {

class ColumnDescriptor;

struct PageLocation {
  /** Offset of the page in the file **/
  int64_t offset;

  /**
   * Size of the page, including header. Sum of compressed_page_size and header
   * length
   */
  int32_t compressed_page_size;

  /**
   * Index within the RowGroup of the first row of the page; this means pages
   * change on record boundaries (r = 0).
   */
  int64_t first_row_index;
  int64_t num_rows;
};

class PARQUET_EXPORT ColumnIndex {
 public:
  virtual ~ColumnIndex() {}

  static std::shared_ptr<ColumnIndex> Make(
      const ColumnDescriptor* descr, const BoundaryOrder::type boundary_order,
      std::vector<bool> null_pages, std::vector<std::string> encoded_min_values,
      std::vector<std::string> encoded_max_values,
      std::vector<PageLocation> page_locations,
      ::arrow::util::optional<std::vector<int64_t>> null_counts = ::arrow::util::nullopt,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// \brief The full type descriptor from the column schema
  virtual const ColumnDescriptor* descr() const = 0;

  /// \brief The physical type of the column schema
  virtual Type::type physical_type() const = 0;

  virtual int64_t num_pages() const = 0;
};

template <typename DType>
class TypedColumnIndex : public ColumnIndex {
 public:
  using T = typename DType::c_type;

  virtual std::shared_ptr<TypedStatistics<DType>> PageStatistics(int64_t page) const = 0;
};

/// \brief Typed version of ColumnIndex::Make
template <typename DType>
std::shared_ptr<TypedColumnIndex<DType>> MakeColumnIndex(
    const ColumnDescriptor* descr, const BoundaryOrder::type boundary_order,
    std::vector<bool> null_pages, std::vector<std::string> encoded_min_values,
    std::vector<std::string> encoded_max_values, std::vector<PageLocation> page_locations,
    ::arrow::util::optional<std::vector<int64_t>> null_counts = ::arrow::util::nullopt,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return std::static_pointer_cast<TypedColumnIndex<DType>>(ColumnIndex::Make(
      descr, boundary_order, std::move(null_pages), std::move(encoded_min_values),
      std::move(encoded_max_values), std::move(page_locations), std::move(null_counts),
      std::move(pool)));
}

}  // namespace parquet