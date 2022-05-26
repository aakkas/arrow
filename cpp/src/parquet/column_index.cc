
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
#include "arrow/util/optional.h"
#include "parquet/encryption/encryption_internal.h"
#include "parquet/encryption/internal_file_encryptor.h"
#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/schema.h"
#include "parquet/statistics.h"
#include "parquet/thrift_internal.h"

namespace parquet {

using ::arrow::util::optional;

class OffsetIndexImpl : public OffsetIndex {
 public:
  OffsetIndexImpl(std::vector<PageLocation> page_locations)
      : page_locations_(std::move(page_locations)) {}

  int64_t num_pages() const override { return page_locations_.size(); }

  const std::vector<PageLocation>& page_locations() const override {
    return page_locations_;
  }

 private:
  std::vector<PageLocation> page_locations_;
};

std::shared_ptr<OffsetIndex> OffsetIndex::Deserialize(
    int64_t total_num_rows, const Buffer& buffer, std::shared_ptr<Decryptor> decryptor) {
  uint32_t len = static_cast<int32_t>(buffer.size());
  format::OffsetIndex offset_index;
  DeserializeThriftMsg(buffer.data(), &len, &offset_index, decryptor);

  std::vector<PageLocation> page_locations(offset_index.page_locations.size());
  auto pl = page_locations.rbegin();
  auto oipl = offset_index.page_locations.rbegin();
  while (oipl != offset_index.page_locations.rend()) {
    pl->offset = oipl->offset;
    pl->compressed_page_size = oipl->compressed_page_size;
    pl->first_row_index = oipl->first_row_index;
    pl->num_rows = total_num_rows - oipl->first_row_index;
    total_num_rows = oipl->first_row_index;
    pl++;
    oipl++;
  }

  return std::make_shared<OffsetIndexImpl>(std::move(page_locations));
}

template <typename DType>
class TypedColumnIndexImpl : public TypedColumnIndex<DType> {
 public:
  using T = typename DType::c_type;

  TypedColumnIndexImpl(const ColumnDescriptor* descr, format::ColumnIndex column_index,
                       ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : descr_(descr), ci_(std::move(column_index)), pool_(pool) {
    boundary_order_ = LoadEnumSafe(&ci_.boundary_order);
    auto pg_count = ci_.null_pages.size();

    if (ci_.max_values.size() != pg_count || ci_.min_values.size() != pg_count ||
        (ci_.__isset.null_counts && ci_.null_counts.size() != pg_count)) {
      throw ParquetException("Can't decode column index. Not all sizes are equal.");
    }
  }

  const ColumnDescriptor* descr() const override { return descr_; }

  BoundaryOrder::type boundary_order() const override { return boundary_order_; }

  int64_t num_pages() const override {
    return static_cast<int64_t>(ci_.null_pages.size());
  }

  std::shared_ptr<Statistics> PageStatistics(int64_t page,
                                             int64_t num_rows) const override {
    return TypedPageStatistics(page, num_rows);
  }

  std::shared_ptr<TypedStatistics<DType>> TypedPageStatistics(
      int64_t page, int64_t num_rows) const override;

 private:
  const ColumnDescriptor* descr_;
  format::ColumnIndex ci_;
  BoundaryOrder::type boundary_order_;
  ::arrow::MemoryPool* pool_;
};

template <typename DType>
std::shared_ptr<TypedStatistics<DType>> TypedColumnIndexImpl<DType>::TypedPageStatistics(
    int64_t page, int64_t num_rows) const {
  bool is_null_page = ci_.null_pages[page];
  bool has_null_count = ci_.__isset.null_counts;

  int64_t num_values = num_rows, null_count = 0;

  if (is_null_page) {
    std::swap(null_count, num_values);
    has_null_count = true;
  } else if (has_null_count) {
    null_count = ci_.null_counts[page];
    num_values -= null_count;
  }

  return MakeStatistics<DType>(descr_, ci_.min_values[page], ci_.max_values[page],
                               num_values, null_count, 0,
                               /*has_min_max*/ !is_null_page, has_null_count,
                               /*has_distinct_count*/ false, pool_);
}

std::shared_ptr<parquet::ColumnIndex> MakeColumnIndex(const ColumnDescriptor* descr,
                                                      format::ColumnIndex column_index) {
#define MAKE_TYPED_COL_INDEX(CAP_TYPE, KLASS) \
  case Type::CAP_TYPE:                        \
    return std::make_shared<TypedColumnIndexImpl<KLASS>>(descr, std::move(column_index))

  switch (descr->physical_type()) {
    MAKE_TYPED_COL_INDEX(BOOLEAN, BooleanType);
    MAKE_TYPED_COL_INDEX(INT32, Int32Type);
    MAKE_TYPED_COL_INDEX(INT64, Int64Type);
    MAKE_TYPED_COL_INDEX(INT96, Int96Type);
    MAKE_TYPED_COL_INDEX(FLOAT, FloatType);
    MAKE_TYPED_COL_INDEX(DOUBLE, DoubleType);
    MAKE_TYPED_COL_INDEX(BYTE_ARRAY, ByteArrayType);
    MAKE_TYPED_COL_INDEX(FIXED_LEN_BYTE_ARRAY, FLBAType);
    default:
      break;
  }
#undef MAKE_TYPED_COL_INDEX

  throw ParquetException("Can't decode column index for selected column type");
}

std::shared_ptr<ColumnIndex> ColumnIndex::Deserialize(
    const ColumnDescriptor* descr, const Buffer& buffer,
    std::shared_ptr<Decryptor> decryptor) {
  uint32_t len = static_cast<int32_t>(buffer.size());

  format::ColumnIndex column_index;
  DeserializeThriftMsg(buffer.data(), &len, &column_index, decryptor);

  return MakeColumnIndex(descr, std::move(column_index));
}

template <typename T, typename TPTR = typename T::TBase>
std::vector<TPTR*> ToPointerVector(std::vector<T>& items) {
  std::vector<TPTR*> pointers;
  pointers.reserve(items.size());

  for (auto& item : items) {
    pointers.push_back(&item);
  }

  return pointers;
}

template <typename IndexBuilderType>
class IndexBuilderImpl : public IndexBuilderType {
 public:
  IndexBuilderImpl(std::shared_ptr<WriterProperties> props, const ColumnDescriptor* descr,
                   int16_t row_group_ordinal, int16_t column_ordinal,
                   std::shared_ptr<Encryptor> encryptor = NULLPTR)
      : props_(std::move(props)),
        descr_(descr),
        row_group_ordinal_(row_group_ordinal),
        column_ordinal_(column_ordinal),
        encryptor_(std::move(encryptor)) {}

  int16_t row_group_ordinal() const override { return row_group_ordinal_; }

  int16_t column_ordinal() const override { return column_ordinal_; }

 protected:
  void UpdateEncryptorAad(int8_t module_type) const {
    if (!encryptor_) {
      return;
    }

    const auto& col_props =
        props_->column_encryption_properties(descr_->path()->ToDotString());
    if (col_props && !col_props->is_encrypted_with_footer_key()) {
      encryptor_->UpdateAad(encryption::CreateModuleAad(
          encryptor_->file_aad(), module_type, row_group_ordinal_, column_ordinal_,
          kNonPageOrdinal));
    }
  }

  std::shared_ptr<WriterProperties> props_;
  const ColumnDescriptor* descr_;
  int row_group_ordinal_;
  int column_ordinal_;
  std::shared_ptr<Encryptor> encryptor_;
};

class ColumnIndexBuilderImpl : public IndexBuilderImpl<ColumnIndexBuilder> {
 public:
  using TBase = ColumnIndexBuilder;

  ColumnIndexBuilderImpl(std::shared_ptr<WriterProperties> props,
                         const ColumnDescriptor* descr, int16_t row_group_ordinal,
                         int16_t column_ordinal,
                         std::shared_ptr<Encryptor> encryptor = NULLPTR)
      : IndexBuilderImpl(std::move(props), descr, row_group_ordinal, column_ordinal,
                         encryptor),
        ascending_(true),
        descending_(true) {
    ci_.__isset.null_counts = true;
  }

  void AddPageStatistics(const EncodedStatistics& stats, bool ascending,
                         bool descending) override {
    ascending_ = ascending_ && ascending;
    descending_ = descending_ && descending;

    if (stats.has_min) {
      ci_.min_values.push_back(stats.min());
      DCHECK(stats.has_max);
      ci_.max_values.push_back(stats.max());
      ci_.null_pages.push_back(false);
    } else {
      ci_.min_values.emplace_back();
      ci_.max_values.emplace_back();
      ci_.null_pages.push_back(true);
    }

    ci_.null_counts.push_back(stats.null_count);
  }

  void Finish() override {
    if (ascending_) {
      ci_.__set_boundary_order(format::BoundaryOrder::type::ASCENDING);
    } else if (descending_) {
      ci_.__set_boundary_order(format::BoundaryOrder::type::DESCENDING);
    } else {
      ci_.__set_boundary_order(format::BoundaryOrder::type::UNORDERED);
    }
  }

  IndexLocation WriteTo(::arrow::io::OutputStream* sink) const override {
    UpdateEncryptorAad(encryption::kColumnIndex);

    IndexLocation location;
    PARQUET_ASSIGN_OR_THROW(location.offset, sink->Tell());

    ThriftSerializer serializer;
    serializer.Serialize(&ci_, sink, encryptor_);

    PARQUET_ASSIGN_OR_THROW(int64_t end, sink->Tell());
    location.length = static_cast<int32_t>(end - location.offset);
    return location;
  }

 private:
  bool ascending_;
  bool descending_;
  format::ColumnIndex ci_;
};

class OffsetIndexBuilderImpl : public IndexBuilderImpl<OffsetIndexBuilder> {
 public:
  using TBase = OffsetIndexBuilder;

  OffsetIndexBuilderImpl(std::shared_ptr<WriterProperties> props,
                         const ColumnDescriptor* descr, int16_t row_group_ordinal,
                         int16_t column_ordinal,
                         std::shared_ptr<Encryptor> encryptor = NULLPTR)
      : IndexBuilderImpl(std::move(props), descr, row_group_ordinal, column_ordinal,
                         encryptor) {}

  void AddPageOffsetInfo(int64_t offset, int32_t compressed_page_size,
                         int64_t first_row_index) override {
    offset_index_.page_locations.emplace_back();
    auto& page_locaton = offset_index_.page_locations.back();
    page_locaton.__set_offset(offset);
    page_locaton.__set_compressed_page_size(compressed_page_size);
    page_locaton.__set_first_row_index(first_row_index);
  }

  void Finish() override {}

  IndexLocation WriteTo(::arrow::io::OutputStream* sink) const override {
    UpdateEncryptorAad(encryption::kOffsetIndex);

    IndexLocation location;
    PARQUET_ASSIGN_OR_THROW(location.offset, sink->Tell());

    ThriftSerializer serializer;
    serializer.Serialize(&offset_index_, sink, encryptor_);

    PARQUET_ASSIGN_OR_THROW(int64_t end, sink->Tell());
    location.length = static_cast<int32_t>(end - location.offset);
    return location;
  }

 private:
  format::OffsetIndex offset_index_;
};

class RowGroupIndexBuilderImpl : public RowGroupIndexBuilder {
 public:
  using TBase = RowGroupIndexBuilder;

  RowGroupIndexBuilderImpl(std::shared_ptr<WriterProperties> props,
                           int16_t row_group_ordinal)
      : props_(std::move(props)), row_group_ordinal_(row_group_ordinal) {}

  ColumnIndexBuilder* ColumnIndex(const ColumnDescriptor* descr, int16_t column_ordinal,
                                  std::shared_ptr<Encryptor> encryptor) override {
    column_index_builders_.emplace_back(props_, descr, row_group_ordinal_, column_ordinal,
                                        encryptor);
    return &column_index_builders_.back();
  }

  OffsetIndexBuilder* OffsetIndex(const ColumnDescriptor* descr, int16_t column_ordinal,
                                  std::shared_ptr<Encryptor> encryptor) override {
    offset_index_builders_.emplace_back(props_, descr, row_group_ordinal_, column_ordinal,
                                        encryptor);
    return &offset_index_builders_.back();
  }

  std::vector<ColumnIndexBuilder*> ColumnIndexes() override {
    return ToPointerVector(column_index_builders_);
  }

  std::vector<OffsetIndexBuilder*> OffsetIndexes() override {
    return ToPointerVector(offset_index_builders_);
  }

 private:
  std::shared_ptr<WriterProperties> props_;
  int16_t row_group_ordinal_;
  std::vector<ColumnIndexBuilderImpl> column_index_builders_;
  std::vector<OffsetIndexBuilderImpl> offset_index_builders_;
};

class FileIndexBuilderImpl : public FileIndexBuilder {
 public:
  FileIndexBuilderImpl(std::shared_ptr<WriterProperties> props)
      : props_(std::move(props)) {}

  RowGroupIndexBuilder* AppendRowGroup() override {
    row_group_builders_.emplace_back(props_,
                                     static_cast<int16_t>(row_group_builders_.size()));
    return &row_group_builders_.back();
  }

  std::vector<RowGroupIndexBuilder*> RowGroups() override {
    return ToPointerVector(row_group_builders_);
  }

 private:
  std::shared_ptr<WriterProperties> props_;
  std::vector<RowGroupIndexBuilderImpl> row_group_builders_;
};

std::unique_ptr<FileIndexBuilder> FileIndexBuilder::Make(
    std::shared_ptr<WriterProperties> props) {
  return std::unique_ptr<FileIndexBuilderImpl>(
      new FileIndexBuilderImpl(std::move(props)));
}

}  // namespace parquet
