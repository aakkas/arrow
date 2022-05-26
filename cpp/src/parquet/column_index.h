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
class Encryptor;
class Decryptor;
class WriterProperties;

struct PARQUET_EXPORT PageLocation {
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

class PARQUET_EXPORT OffsetIndex {
 public:
  virtual ~OffsetIndex() {}

  static std::shared_ptr<OffsetIndex> Deserialize(
      int64_t total_num_rows, const Buffer& buffer,
      std::shared_ptr<Decryptor> decryptor = NULLPTR);

  virtual int64_t num_pages() const = 0;
  virtual const std::vector<PageLocation>& page_locations() const = 0;
};

class PARQUET_EXPORT ColumnIndex {
 public:
  virtual ~ColumnIndex() {}

  static std::shared_ptr<ColumnIndex> Deserialize(
      const ColumnDescriptor* descr, const Buffer& buffer,
      std::shared_ptr<Decryptor> decryptor = NULLPTR);

  /// \brief The full type descriptor from the column schema
  virtual const ColumnDescriptor* descr() const = 0;

  virtual BoundaryOrder::type boundary_order() const = 0;

  virtual int64_t num_pages() const = 0;

  virtual std::shared_ptr<Statistics> PageStatistics(
      int64_t page, int64_t num_rows) const = 0;
};

template <typename DType>
class PARQUET_EXPORT TypedColumnIndex : public ColumnIndex {
 public:
  using T = typename DType::c_type;

  virtual std::shared_ptr<TypedStatistics<DType>> TypedPageStatistics(
      int64_t page, int64_t num_rows) const = 0;
};

struct IndexLocation {
  int64_t offset;
  int32_t length;
};

class PARQUET_EXPORT IndexBuilder {
 public:
  virtual ~IndexBuilder() {}

  virtual void Finish() = 0;

  virtual IndexLocation WriteTo(::arrow::io::OutputStream* dst) const = 0;

  virtual int16_t row_group_ordinal() const = 0;
  virtual int16_t column_ordinal() const = 0;
};

class PARQUET_EXPORT OffsetIndexBuilder : public IndexBuilder {
 public:
  virtual ~OffsetIndexBuilder() {}

  virtual void AddPageOffsetInfo(int64_t offset, int32_t compressed_page_size,
                                 int64_t first_row_index) = 0;
};

class PARQUET_EXPORT ColumnIndexBuilder : public IndexBuilder {
 public:
  virtual ~ColumnIndexBuilder() {}

  virtual void AddPageStatistics(const EncodedStatistics& stats, bool ascending,
                                 bool descending) = 0;
};

class PARQUET_EXPORT RowGroupIndexBuilder {
 public:
  virtual ~RowGroupIndexBuilder() {}
  virtual ColumnIndexBuilder* ColumnIndex(
      const ColumnDescriptor* descr, int16_t column_ordinal,
      std::shared_ptr<Encryptor> encryptor = NULLPTR) = 0;
  virtual OffsetIndexBuilder* OffsetIndex(
      const ColumnDescriptor* descr, int16_t column_ordinal,
      std::shared_ptr<Encryptor> encryptor = NULLPTR) = 0;

  virtual std::vector<ColumnIndexBuilder*> ColumnIndexes() = 0;
  virtual std::vector<OffsetIndexBuilder*> OffsetIndexes() = 0;
};

class PARQUET_EXPORT FileIndexBuilder {
 public:
  virtual ~FileIndexBuilder() {}

  static std::unique_ptr<FileIndexBuilder> Make(std::shared_ptr<WriterProperties> props);

  virtual RowGroupIndexBuilder* AppendRowGroup() = 0;

  virtual std::vector<RowGroupIndexBuilder*> RowGroups() = 0;
};

}  // namespace parquet