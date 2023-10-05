// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/dataset/file_parquet.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression_internal.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/scanner.h"
#include "arrow/filesystem/path_util.h"
#include "arrow/scalar.h"
#include "arrow/table.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/util/mutex.h"
#include "arrow/util/range.h"
#include "arrow/util/row_ranges.h"
#include "arrow/util/tracing_internal.h"
#include "arrow/util/variant.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"
#include "parquet/arrow/writer.h"
#include "parquet/column_index.h"
#include "parquet/file_reader.h"
#include "parquet/properties.h"
#include "parquet/statistics.h"

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;
using internal::Iota;

namespace dataset {

using parquet::arrow::SchemaField;
using parquet::arrow::SchemaManifest;
using parquet::arrow::StatisticsAsScalars;

namespace {

parquet::ReaderProperties MakeReaderProperties(
    const ParquetFileFormat& format, ParquetFragmentScanOptions* parquet_scan_options,
    MemoryPool* pool = default_memory_pool()) {
  // Can't mutate pool after construction
  parquet::ReaderProperties properties(pool);
  if (parquet_scan_options->reader_properties->is_buffered_stream_enabled()) {
    properties.enable_buffered_stream();
  } else {
    properties.disable_buffered_stream();
  }
  properties.set_buffer_size(parquet_scan_options->reader_properties->buffer_size());
  properties.file_decryption_properties(
      parquet_scan_options->reader_properties->file_decryption_properties());
  properties.set_thrift_string_size_limit(
      parquet_scan_options->reader_properties->thrift_string_size_limit());
  properties.set_thrift_container_size_limit(
      parquet_scan_options->reader_properties->thrift_container_size_limit());
  return properties;
}

parquet::ArrowReaderProperties MakeArrowReaderProperties(
    const ParquetFileFormat& format, const parquet::FileMetaData& metadata) {
  parquet::ArrowReaderProperties properties(/* use_threads = */ false);
  for (const std::string& name : format.reader_options.dict_columns) {
    auto column_index = metadata.schema()->ColumnIndex(name);
    properties.set_read_dictionary(column_index, true);
  }
  properties.set_coerce_int96_timestamp_unit(
      format.reader_options.coerce_int96_timestamp_unit);
  return properties;
}

parquet::ArrowReaderProperties MakeArrowReaderProperties(
    const ParquetFileFormat& format, const parquet::FileMetaData& metadata,
    const ScanOptions& options, const ParquetFragmentScanOptions& parquet_scan_options) {
  auto arrow_properties = MakeArrowReaderProperties(format, metadata);
  arrow_properties.set_batch_size(options.batch_size);
  // Must be set here since the sync ScanTask handles pre-buffering itself
  arrow_properties.set_pre_buffer(
      parquet_scan_options.arrow_reader_properties->pre_buffer());
  arrow_properties.set_cache_options(
      parquet_scan_options.arrow_reader_properties->cache_options());
  arrow_properties.set_io_context(
      parquet_scan_options.arrow_reader_properties->io_context());
  arrow_properties.set_use_threads(options.use_threads);
  return arrow_properties;
}

Result<std::shared_ptr<SchemaManifest>> GetSchemaManifest(
    const parquet::FileMetaData& metadata,
    const parquet::ArrowReaderProperties& properties) {
  auto manifest = std::make_shared<SchemaManifest>();
  const std::shared_ptr<const ::arrow::KeyValueMetadata>& key_value_metadata =
      metadata.key_value_metadata();
  RETURN_NOT_OK(SchemaManifest::Make(metadata.schema(), key_value_metadata, properties,
                                     manifest.get()));
  return manifest;
}

bool IsNan(const Scalar& value) {
  if (value.is_valid) {
    if (value.type->id() == Type::FLOAT) {
      const FloatScalar& float_scalar = checked_cast<const FloatScalar&>(value);
      return std::isnan(float_scalar.value);
    } else if (value.type->id() == Type::DOUBLE) {
      const DoubleScalar& double_scalar = checked_cast<const DoubleScalar&>(value);
      return std::isnan(double_scalar.value);
    }
  }
  return false;
}

std::optional<compute::Expression> ColumnChunkStatisticsAsExpression(
    const SchemaField& schema_field, const parquet::RowGroupMetaData& metadata) {
  // For the remaining of this function, failure to extract/parse statistics
  // are ignored by returning nullptr. The goal is two fold. First
  // avoid an optimization which breaks the computation. Second, allow the
  // following columns to maybe succeed in extracting column statistics.

util::optional<compute::Expression> StatisticsAsExpression(
    const SchemaField& schema_field,
    const std::shared_ptr<parquet::Statistics>& statistics) {
  // For now, only leaf (primitive) types are supported.
  if (!schema_field.is_leaf()) {
    return std::nullopt;
  }

  auto column_metadata = metadata.ColumnChunk(schema_field.column_index);
  auto statistics = column_metadata->statistics();
  if (statistics == nullptr) {
    return util::nullopt;
  }

  const auto& field = schema_field.field;

  if (statistics == nullptr) {
    return std::nullopt;
  }

  return ParquetFileFragment::EvaluateStatisticsAsExpression(*field, *statistics);
}

util::optional<compute::Expression> ColumnChunkStatisticsAsExpression(
    const SchemaField& schema_field, const parquet::RowGroupMetaData& metadata) {
  // For the remaining of this function, failure to extract/parse statistics
  // are ignored by returning nullptr. The goal is two fold. First
  // avoid an optimization which breaks the computation. Second, allow the
  // following columns to maybe succeed in extracting column statistics.

  // For now, only leaf (primitive) types are supported.
  if (!schema_field.is_leaf()) {
    return util::nullopt;
  }

  auto column_metadata = metadata.ColumnChunk(schema_field.column_index);
  return StatisticsAsExpression(schema_field, column_metadata->statistics());
}

void AddColumnIndices(const SchemaField& schema_field,
                      std::vector<int>* column_projection) {
  if (schema_field.is_leaf()) {
    column_projection->push_back(schema_field.column_index);
  } else {
    // The following ensure that complex types, e.g. struct,  are materialized.
    for (const auto& child : schema_field.children) {
      AddColumnIndices(child, column_projection);
    }
  }
}

Status ResolveOneFieldRef(
    const SchemaManifest& manifest, const FieldRef& field_ref,
    const std::unordered_map<std::string, const SchemaField*>& field_lookup,
    const std::unordered_set<std::string>& duplicate_fields,
    std::vector<int>* columns_selection) {
  if (const std::string* name = field_ref.name()) {
    auto it = field_lookup.find(*name);
    if (it != field_lookup.end()) {
      AddColumnIndices(*it->second, columns_selection);
    } else if (duplicate_fields.find(*name) != duplicate_fields.end()) {
      // We shouldn't generally get here because SetProjection will reject such references
      return Status::Invalid("Ambiguous reference to column '", *name,
                             "' which occurs more than once");
    }
    // "Virtual" column: field is not in file but is in the ScanOptions.
    // Ignore it here, as projection will pad the batch with a null column.
    return Status::OK();
  }

  const SchemaField* toplevel = nullptr;
  const SchemaField* field = nullptr;
  if (const std::vector<FieldRef>* refs = field_ref.nested_refs()) {
    // Only supports a sequence of names
    for (const auto& ref : *refs) {
      if (const std::string* name = ref.name()) {
        if (!field) {
          // First lookup, top-level field
          auto it = field_lookup.find(*name);
          if (it != field_lookup.end()) {
            field = it->second;
            toplevel = field;
          } else if (duplicate_fields.find(*name) != duplicate_fields.end()) {
            return Status::Invalid("Ambiguous reference to column '", *name,
                                   "' which occurs more than once");
          } else {
            // Virtual column
            return Status::OK();
          }
        } else {
          const SchemaField* result = nullptr;
          for (const auto& child : field->children) {
            if (child.field->name() == *name) {
              if (!result) {
                result = &child;
              } else {
                return Status::Invalid("Ambiguous nested reference to column '", *name,
                                       "' which occurs more than once in field ",
                                       field->field->ToString());
              }
            }
          }
          if (!result) {
            // Virtual column
            return Status::OK();
          }
          field = result;
        }
        continue;
      }
      return Status::NotImplemented("Inferring column projection from FieldRef ",
                                    field_ref.ToString());
    }
  } else {
    return Status::NotImplemented("Inferring column projection from FieldRef ",
                                  field_ref.ToString());
  }

  if (field) {
    // TODO(ARROW-1888): support fine-grained column projection. We should be
    // able to materialize only the child fields requested, and not the entire
    // top-level field.
    // Right now, if enabled, projection/filtering will fail when they cast the
    // physical schema to the dataset schema.
    AddColumnIndices(*toplevel, columns_selection);
  }
  return Status::OK();
}

// Converts a field ref into a position-independent ref (containing only a sequence of
// names) based on the dataset schema. Returns `false` if no conversion was needed.
Result<FieldRef> MaybeConvertFieldRef(FieldRef ref, const Schema& dataset_schema) {
  if (ARROW_PREDICT_TRUE(ref.IsNameSequence())) {
    return std::move(ref);
  }

  ARROW_ASSIGN_OR_RAISE(auto path, ref.FindOne(dataset_schema));
  std::vector<FieldRef> named_refs;
  named_refs.reserve(path.indices().size());

  const FieldVector* child_fields = &dataset_schema.fields();
  for (auto index : path) {
    const auto& child_field = *(*child_fields)[index];
    named_refs.emplace_back(child_field.name());
    child_fields = &child_field.type()->fields();
  }

  return named_refs.size() == 1 ? std::move(named_refs[0])
                                : FieldRef(std::move(named_refs));
}

// Compute the column projection based on the scan options
Result<std::vector<int>> InferColumnProjection(
    const parquet::arrow::SchemaManifest& manifest,
    const std::vector<FieldRef>& field_refs) {
  // Build a lookup table from top level field name to field metadata.
  // This is to avoid quadratic-time mapping of projected fields to
  // column indices, in the common case of selecting top level
  // columns. For nested fields, we will pay the cost of a linear scan
  // assuming for now that this is relatively rare, but this can be
  // optimized. (Also, we don't want to pay the cost of building all
  // the lookup tables up front if they're rarely used.)
  std::unordered_map<std::string, const SchemaField*> field_lookup;
  std::unordered_set<std::string> duplicate_fields;
  for (const auto& schema_field : manifest.schema_fields) {
    const auto it = field_lookup.emplace(schema_field.field->name(), &schema_field);
    if (!it.second) {
      duplicate_fields.emplace(schema_field.field->name());
    }
  }

  std::vector<int> columns_selection;
  for (auto& ref : field_refs) {
    // In the (unlikely) absence of a known dataset schema, we require that all
    // materialized refs are named.
    if (options.dataset_schema) {
      ARROW_ASSIGN_OR_RAISE(
          ref, MaybeConvertFieldRef(std::move(ref), *options.dataset_schema));
    }
    RETURN_NOT_OK(ResolveOneFieldRef(manifest, ref, field_lookup, duplicate_fields,
                                     &columns_selection));
  }
  return columns_selection;
}

// Compute the column projection based on the scan options
Result<std::vector<int>> InferColumnProjection(const parquet::arrow::FileReader& reader,
                                               const ScanOptions& options) {
  // Checks if the field is needed in either the projection or the filter.
  auto manifest = reader.manifest();
  return InferColumnProjection(manifest, options.MaterializedFields());
}

// Compute the column projection based on the scan options
Result<std::vector<int>> InferColumnProjection(
    const Schema& physical_schema, const parquet::arrow::SchemaManifest& manifest,
    const compute::Expression& predicate, const bool only_leafs = false) {
  // Checks if the field is needed in either the projection or the filter.

  auto field_refs = FieldsInExpression(predicate);
  if (only_leafs) {
    std::vector<int> result;
    std::unordered_set<int> visited;
    for (const auto& field_ref : field_refs) {
      ARROW_ASSIGN_OR_RAISE(auto match, field_ref.FindOneOrNone(physical_schema));
      if (visited.insert(match[0]).second) {
        const SchemaField& schema_field = manifest.schema_fields[match[0]];

        if (schema_field.is_leaf()) {
          result.push_back(schema_field.column_index);
        }
      }
    }

    return result;
  }

  return InferColumnProjection(manifest, field_refs);
}

Status WrapSourceError(const Status& status, const std::string& path) {
  return status.WithMessage("Could not open Parquet input source '", path,
                            "': ", status.message());
}

Result<bool> IsSupportedParquetFile(const ParquetFileFormat& format,
                                    const FileSource& source) {
  BEGIN_PARQUET_CATCH_EXCEPTIONS
  try {
    ARROW_ASSIGN_OR_RAISE(auto input, source.Open());
    ARROW_ASSIGN_OR_RAISE(
        auto parquet_scan_options,
        GetFragmentScanOptions<ParquetFragmentScanOptions>(
            kParquetTypeName, nullptr, format.default_fragment_scan_options));
    auto reader = parquet::ParquetFileReader::Open(
        std::move(input), MakeReaderProperties(format, parquet_scan_options.get()));
    std::shared_ptr<parquet::FileMetaData> metadata = reader->metadata();
    return metadata != nullptr && metadata->can_decompress();
  } catch (const ::parquet::ParquetInvalidOrCorruptedFileException& e) {
    ARROW_UNUSED(e);
    return false;
  }
  END_PARQUET_CATCH_EXCEPTIONS
}

}  // namespace

std::optional<compute::Expression> ParquetFileFragment::EvaluateStatisticsAsExpression(
    const Field& field, const parquet::Statistics& statistics) {
  auto field_expr = compute::field_ref(field.name());

  // Optimize for corner case where all values are nulls
  if (statistics.num_values() == 0 && statistics.null_count() > 0) {
    return is_null(std::move(field_expr));
  }

  std::shared_ptr<Scalar> min, max;
  if (!StatisticsAsScalars(statistics, &min, &max).ok()) {
    return std::nullopt;
  }

  auto maybe_min = min->CastTo(field.type());
  auto maybe_max = max->CastTo(field.type());

  if (maybe_min.ok() && maybe_max.ok()) {
    min = maybe_min.MoveValueUnsafe();
    max = maybe_max.MoveValueUnsafe();

    if (min->Equals(*max)) {
      auto single_value = compute::equal(field_expr, compute::literal(std::move(min)));

      if (statistics.null_count() == 0) {
        return single_value;
      }
      return compute::or_(std::move(single_value), is_null(std::move(field_expr)));
    }

    auto lower_bound = compute::greater_equal(field_expr, compute::literal(min));
    auto upper_bound = compute::less_equal(field_expr, compute::literal(max));
    compute::Expression in_range;

    // Since the minimum & maximum values are NaN, useful statistics
    // cannot be extracted for checking the presence of a value within
    // range
    if (IsNan(*min) && IsNan(*max)) {
      return std::nullopt;
    }

    // If either minimum or maximum is NaN, it should be ignored for the
    // range computation
    if (IsNan(*min)) {
      in_range = std::move(upper_bound);
    } else if (IsNan(*max)) {
      in_range = std::move(lower_bound);
    } else {
      in_range = compute::and_(std::move(lower_bound), std::move(upper_bound));
    }

    if (statistics.null_count() != 0) {
      return compute::or_(std::move(in_range), compute::is_null(field_expr));
    }
    return in_range;
  }
  return std::nullopt;
}

ParquetFileFormat::ParquetFileFormat()
    : FileFormat(std::make_shared<ParquetFragmentScanOptions>()) {}
class FilterResult {
 public:
  struct RangeResult {
    util::RowRanges include;
    util::RowRanges is_null;

    RangeResult() {}
    RangeResult(util::RowRanges include) : include(std::move(include)) {}
    RangeResult(util::RowRanges include, util::RowRanges is_null)
        : include(std::move(include)), is_null(std::move(is_null)) {}

    Result<RangeResult> Invert() const { return RangeResult{include.Invert(), is_null}; }

    Result<RangeResult> IsNull() const { return RangeResult{is_null}; }

    Result<RangeResult> IsValid() const { return is_null.Invert(); }

    Result<RangeResult> TrueUnlessNull() const {
      return RangeResult{is_null.Invert(), is_null.Invert()};
    }

    Result<RangeResult> And(const RangeResult& rhs) const {
      return RangeResult{include.Intersect(rhs.include), is_null.Union(rhs.is_null)};
    }

    Result<RangeResult> AndKleene(const RangeResult& rhs) const {
      ARROW_ASSIGN_OR_RAISE(auto lhs_valid, IsValid());
      auto lhs_true = lhs_valid.include.Intersect(include);
      auto lhs_false = lhs_valid.include.Intersect(include.Invert());

      ARROW_ASSIGN_OR_RAISE(auto rhs_valid, rhs.IsValid());
      auto rhs_true = rhs_valid.include.Intersect(rhs.include);
      auto rhs_false = rhs_valid.include.Intersect(rhs.include.Invert());

      auto result_include = lhs_true.Intersect(rhs_true);
      auto result_is_null = lhs_false.Union(rhs_false).Union(result_include).Invert();
      return RangeResult{std::move(result_include), std::move(result_is_null)};
    }

    Result<RangeResult> AndNot(const RangeResult& rhs) const {
      ARROW_ASSIGN_OR_RAISE(auto not_rhs, rhs.Invert());
      return And(not_rhs);
    }

    Result<RangeResult> AndNotKleene(const RangeResult& rhs) const {
      ARROW_ASSIGN_OR_RAISE(auto not_rhs, rhs.Invert());
      return AndKleene(not_rhs);
    }

    Result<RangeResult> Or(const RangeResult& rhs) const {
      return RangeResult{include.Union(rhs.include), is_null.Union(rhs.is_null)};
    }

    Result<RangeResult> OrKleene(const RangeResult& rhs) const {
      ARROW_ASSIGN_OR_RAISE(auto lhs_valid, IsValid());
      auto lhs_true = lhs_valid.include.Intersect(include);
      auto lhs_false = lhs_valid.include.Intersect(include.Invert());

      ARROW_ASSIGN_OR_RAISE(auto rhs_valid, rhs.IsValid());
      auto rhs_true = rhs_valid.include.Intersect(rhs.include);
      auto rhs_false = rhs_valid.include.Intersect(rhs.include.Invert());

      auto result_include = lhs_true.Union(rhs_true);
      auto result_is_null = result_include.Union(lhs_false.Intersect(rhs_false)).Invert();

      return RangeResult{std::move(result_include), std::move(result_is_null)};
    }

    Result<RangeResult> Xor(const RangeResult& rhs) const {
      ARROW_ASSIGN_OR_RAISE(auto lhs_and_not_rhs, AndNot(rhs));
      ARROW_ASSIGN_OR_RAISE(auto rhs_and_not_lhs, rhs.AndNot(*this));
      return lhs_and_not_rhs.Or(rhs_and_not_lhs);
    }

    static const RangeResult& NONE() {
      static RangeResult none;
      return none;
    }

    static const RangeResult& ALL() {
      static RangeResult all{util::RowRanges::ALL()};
      return all;
    }

    static const RangeResult& ALL_NULLABLE() {
      static RangeResult all_nullable{util::RowRanges::ALL(), util::RowRanges::ALL()};
      return all_nullable;
    }
  };

  struct DatumResult {
    Datum datum;

    Result<RangeResult> Truthy() const { return RangeResult::ALL(); }
  };

  struct ColumnResult {
    Type::type type;
    const SchemaField* schema_field;

    Result<RangeResult> Truthy() const { return RangeResult::ALL(); }

    Result<RangeResult> Compare(const DatumResult& datum,
                                compute::Comparison::type cmp) const {
      return RangeResult::ALL();
    }
  };

  FilterResult() = default;
  FilterResult(ColumnResult column) : impl_(std::make_shared<Impl>(std::move(column))) {}
  FilterResult(RangeResult range) : impl_(std::make_shared<Impl>(std::move(range))) {}
  FilterResult(DatumResult datum) : impl_(std::make_shared<Impl>(std::move(datum))) {}
  FilterResult(Datum datum)
      : impl_(std::make_shared<Impl>(DatumResult{std::move(datum)})) {}

  const ColumnResult* column() const { return util::get_if<ColumnResult>(impl_.get()); }
  const RangeResult* range() const { return util::get_if<RangeResult>(impl_.get()); }
  const DatumResult* datum() const { return util::get_if<DatumResult>(impl_.get()); }

  Result<RangeResult> Truthy() const {
    if (auto r = range()) {
      return *r;
    } else if (auto d = datum()) {
      return d->Truthy();
    } else if (auto c = column()) {
      return c->Truthy();
    }

    return Status::Invalid("Invalid FilterResult");
  }

  static Result<FilterResult> CallFunction(const std::string& function_name,
                                           const std::vector<FilterResult>& arguments);

  typedef Result<FilterResult> (*FilterFunction)(const std::vector<FilterResult>& args);
  static FilterFunction GetFunction(const std::string& function_name);

 private:
  using Impl = util::Variant<ColumnResult, RangeResult, DatumResult>;
  std::shared_ptr<Impl> impl_;
};

Result<FilterResult> FilterResult::CallFunction(
    const std::string& function_name, const std::vector<FilterResult>& arguments) {
  if (auto func = GetFunction(function_name)) {
    return func(arguments);
  }

  return RangeResult::ALL();
}

#define TRUTHY(lhs, args, idx)                                      \
  if (idx >= args.size()) {                                         \
    return Status::IndexError("Insufficient number of arguments."); \
  }                                                                 \
  ARROW_ASSIGN_OR_RAISE(lhs, args[idx].Truthy());

static Result<FilterResult> Invert(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& truthy, args, 0);
  return truthy.Invert();
}

static Result<FilterResult> IsNull(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& truthy, args, 0);
  return truthy.IsNull();
}

static Result<FilterResult> IsValid(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& truthy, args, 0);
  return truthy.IsValid();
}

static Result<FilterResult> TrueUnlessNull(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& truthy, args, 0);
  return truthy.TrueUnlessNull();
}

static Result<FilterResult> And(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.And(rhs);
}

static Result<FilterResult> AndKleene(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.AndKleene(rhs);
}

static Result<FilterResult> AndNot(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.AndNot(rhs);
}

static Result<FilterResult> AndNotKleene(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.AndNotKleene(rhs);
}

static Result<FilterResult> Or(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.Or(rhs);
}

static Result<FilterResult> OrKleene(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.OrKleene(rhs);
}

static Result<FilterResult> Xor(const std::vector<FilterResult>& args) {
  TRUTHY(const auto& lhs, args, 0);
  TRUTHY(const auto& rhs, args, 1);
  return lhs.Xor(rhs);
}

#undef TRUTHY

static Result<FilterResult> Compare(const std::vector<FilterResult>& args,
                                    const compute::Comparison::type cmp) {
  if (args.size() < 2) {
    return Status::IndexError("Insufficient number of arguments.");
  }

  const auto& lhs = args[0];
  const auto& rhs = args[1];

  if (auto c = lhs.column()) {
    if (auto d = rhs.datum()) {
      return c->Compare(*d, cmp);
    }
  } else if (auto d = lhs.datum()) {
    if (auto c = rhs.column()) {
      return c->Compare(*d, compute::Comparison::GetFlipped(cmp));
    }
  }

  return FilterResult::RangeResult::ALL();
}

static Result<FilterResult> Equal(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::EQUAL);
}

static Result<FilterResult> NotEqual(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::NOT_EQUAL);
}

static Result<FilterResult> Less(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::LESS);
}

static Result<FilterResult> LessEqual(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::LESS_EQUAL);
}

static Result<FilterResult> Greater(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::GREATER);
}

static Result<FilterResult> GreaterEqual(const std::vector<FilterResult>& args) {
  return Compare(args, compute::Comparison::type::GREATER_EQUAL);
}

FilterResult::FilterFunction FilterResult::GetFunction(const std::string& function_name) {
  static std::unordered_map<std::string, FilterFunction> functions{
      {"is_null", IsNull},
      {"is_valid", IsValid},
      {"true_unless_null", TrueUnlessNull},
      // Logical
      {"invert", Invert},
      {"and", And},
      {"and_kleene", AndKleene},
      {"and_not", AndNot},
      {"and_not_kleene", AndNotKleene},
      {"or", Or},
      {"or_kleene", OrKleene},
      {"xor", Xor},
      // Comparison
      {"equal", Equal},
      {"not_equal", NotEqual},
      {"less", Less},
      {"less_equal", LessEqual},
      {"greater", Greater},
      {"greater_equal", GreaterEqual},
  };

  auto it = functions.find(function_name);
  if (it != functions.end()) {
    return it->second;
  }

  return nullptr;
}

//
// ParquetColumnIndexProvider
//

class ParquetColumnIndexProviderImpl : public ParquetColumnIndexProvider {
 public:
  ParquetColumnIndexProviderImpl(std::shared_ptr<ParquetFileFormat> parquet_format,
                                 FileSource metadata_source,
                                 std::shared_ptr<parquet::FileMetaData> metadata,
                                 std::shared_ptr<Schema> physical_schema,
                                 std::shared_ptr<parquet::arrow::SchemaManifest> manifest)
      : parquet_format_(std::move(parquet_format)),
        metadata_source_(std::move(metadata_source)),
        metadata_(std::move(metadata)),
        physical_schema_(std::move(physical_schema)),
        manifest_(std::move(manifest)) {}

  static std::shared_ptr<ParquetColumnIndexProvider> Make(
      std::shared_ptr<ParquetFileFormat> parquet_format, FileSource metadata_source,
      std::shared_ptr<parquet::FileMetaData> metadata,
      std::shared_ptr<Schema> physical_schema,
      std::shared_ptr<parquet::arrow::SchemaManifest> manifest) {
    return std::make_shared<ParquetColumnIndexProviderImpl>(
        std::move(parquet_format), std::move(metadata_source), std::move(metadata),
        std::move(physical_schema), std::move(manifest));
  }

  std::shared_ptr<parquet::ColumnIndex> GetColumnIndex(int row_group,
                                                       int column) const override {
    auto lock = mutex_.Lock();
    return GetColumnIndex_(row_group, column);
  }

  Result<bool> HasColumnIndexes(const compute::Expression& predicate,
                                const std::vector<int>& row_groups) const override {
    if (row_groups.empty()) {
      return true;
    }

    ARROW_ASSIGN_OR_RAISE(auto columns,
                          InferColumnProjection(*physical_schema_, *manifest_, predicate,
                                                /*only_leafs*/ true));

    return HasColumnIndexes(columns, row_groups);
  }

  bool HasColumnIndexes(const std::vector<int>& columns,
                        const std::vector<int>& row_groups) const override {
    if (row_groups.empty() || columns.empty()) {
      return true;
    }

    auto lock = mutex_.Lock();
    for (const auto r : row_groups) {
      if (!HasColumnIndexes_(r, columns)) {
        return false;
      }
    }

    return true;
  }

  const Status EnsureCompleteColumnIndexes(const compute::Expression& predicate,
                                           const std::vector<int>& row_groups) override {
    if (row_groups.empty()) {
      return Status::OK();
    }

    ARROW_ASSIGN_OR_RAISE(auto columns,
                          InferColumnProjection(*physical_schema_, *manifest_, predicate,
                                                /*only_leafs*/ true));

    return EnsureCompleteColumnIndexes(columns, row_groups);
  }

  const Status EnsureCompleteColumnIndexes(const std::vector<int>& columns,
                                           const std::vector<int>& row_groups) override {
    if (HasColumnIndexes(columns, row_groups)) {
      return Status::OK();
    }

    auto scan_options = std::make_shared<ScanOptions>();
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          parquet_format_->GetReader(metadata_source_, scan_options));

    for (const auto r : row_groups) {
      ReadColumnIndexes(reader, r, columns);
    }

    return Status::OK();
  }

 private:
  const std::shared_ptr<parquet::ColumnIndex> GetColumnIndex_(const int row_group,
                                                              const int column) const {
    const auto& it_rg = column_indexes_.find(row_group);
    if (it_rg != column_indexes_.end()) {
      const auto& it_c = it_rg->second.find(column);
      if (it_c != it_rg->second.end()) {
        return it_c->second;
      }
    }

    return nullptr;
  }

  bool HasColumnIndexes_(const int row_group, const std::vector<int>& columns) const {
    for (const auto c : columns) {
      if (!GetColumnIndex_(row_group, c)) {
        return false;
      }
    }

    return false;
  }

  void ReadColumnIndexes(const std::shared_ptr<parquet::arrow::FileReader> reader,
                         const int row_group, const std::vector<int>& columns) {
    if (!HasColumnIndexes({row_group}, columns)) {
      auto row_group_reader = reader->parquet_reader()->RowGroup(row_group);

      for (const auto c : columns) {
        if (!GetColumnIndex(row_group, c)) {
          auto column_index = row_group_reader->ReadColumnIndex(c);

          auto lock = mutex_.Lock();
          column_indexes_[row_group][c] = std::move(column_index);
        }
      }
    }
  }

  inline Result<const SchemaField*> GetSchemaField(const FieldRef& field_ref) {
    ARROW_ASSIGN_OR_RAISE(auto field_path, field_ref.FindOneOrNone(*physical_schema_));
    if (field_path.empty()) {
      return Status::Invalid("Could not find schema field for ", field_ref.ToString());
    }

    return &manifest_->schema_fields[field_path[0]];
  }

  Result<FilterResult> GetRowRanges(const compute::Expression& expr, const int row_group,
                                    const int64_t row_count,
                                    compute::ExecContext* exec_context) {
    if (exec_context == nullptr) {
      compute::ExecContext exec_context;
      return GetRowRanges(expr, row_group, row_count, &exec_context);
    }

    if (!expr.IsBound()) {
      return Status::Invalid("Cannot Execute unbound expression.");
    }

    if (!expr.IsScalarExpression()) {
      return Status::Invalid(
          "ExecuteScalarExpression cannot Execute non-scalar expression ",
          expr.ToString());
    }

    if (auto lit = expr.literal()) {
      return FilterResult::DatumResult{*lit};
    }

    if (auto param = expr.parameter()) {
      const auto schema_field = manifest_
                              ->Get(FieldPath(std::vector<int>(param->indices.begin(),
                                                               param->indices.end())))
                              .ValueOr(nullptr);

      return FilterResult::ColumnResult{param->descr.type->id(), schema_field};
    }

    auto call = compute::CallNotNull(expr);

    std::vector<FilterResult> arguments(call->arguments.size());
    bool all_datum = true;
    for (size_t i = 0; i < arguments.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(arguments[i], GetRowRanges(call->arguments[i], row_group,
                                                       row_count, exec_context));
      all_datum = all_datum && arguments[i].datum() != nullptr;
    }

    if (!all_datum) {
      return FilterResult::CallFunction(call->function_name, arguments);
    }

    DCHECK(all_datum);
    std::vector<Datum> datum_args;
    datum_args.reserve(arguments.size());
    for (size_t i = 0; i < arguments.size(); i++) {
      datum_args[i] = std::move(arguments[i].datum()->datum);
    }

    auto executor = compute::detail::KernelExecutor::MakeScalar();

    compute::KernelContext kernel_context(exec_context);
    kernel_context.SetState(call->kernel_state.get());

    auto kernel = call->kernel;
    auto descrs = compute::GetDescriptors(datum_args);
    auto options = call->options.get();
    RETURN_NOT_OK(executor->Init(&kernel_context, {kernel, descrs, options}));

    compute::detail::DatumAccumulator listener;
    RETURN_NOT_OK(executor->Execute(datum_args, &listener));
    const auto out = executor->WrapResults(datum_args, listener.values());
#ifndef NDEBUG
    DCHECK_OK(executor->CheckResultType(out, call->function_name.c_str()));
#endif

    return out;
  }

  std::shared_ptr<ParquetFileFormat> parquet_format_;
  FileSource metadata_source_;
  std::shared_ptr<parquet::FileMetaData> metadata_;
  std::shared_ptr<Schema> physical_schema_;
  std::shared_ptr<parquet::arrow::SchemaManifest> manifest_;
  std::unordered_map<int, std::unordered_map<int, std::shared_ptr<parquet::ColumnIndex>>>
      column_indexes_;
  mutable util::Mutex mutex_;
};

class ParquetColumnIndexProviderView : public ParquetColumnIndexProvider {
 public:
  ParquetColumnIndexProviderView(std::shared_ptr<ParquetColumnIndexProvider> provider,
                                 std::vector<int> row_group_mapping)
      : provider_(std::move(provider)),
        row_group_mapping_(std::move(row_group_mapping)) {}

  static std::shared_ptr<ParquetColumnIndexProvider> Make(
      std::shared_ptr<ParquetColumnIndexProvider> provider,
      std::vector<int> row_group_mapping) {
    return std::make_shared<ParquetColumnIndexProviderView>(std::move(provider),
                                                            std::move(row_group_mapping));
  }

  std::shared_ptr<parquet::ColumnIndex> GetColumnIndex(int row_group,
                                                       int column) const override {
    return provider_->GetColumnIndex(TranslateRowGroupId(row_group), column);
  }

  Result<bool> HasColumnIndexes(const compute::Expression& predicate,
                                const std::vector<int>& row_groups) const override {
    return provider_->HasColumnIndexes(predicate, TranslateRowGroupIds(row_groups));
  }

  bool HasColumnIndexes(const std::vector<int>& columns,
                        const std::vector<int>& row_groups) const override {
    return provider_->HasColumnIndexes(columns, TranslateRowGroupIds(row_groups));
  }

  const Status EnsureCompleteColumnIndexes(const compute::Expression& predicate,
                                           const std::vector<int>& row_groups) override {
    return provider_->EnsureCompleteColumnIndexes(predicate,
                                                  TranslateRowGroupIds(row_groups));
  }

  const Status EnsureCompleteColumnIndexes(const std::vector<int>& columns,
                                           const std::vector<int>& row_groups) override {
    return provider_->EnsureCompleteColumnIndexes(columns,
                                                  TranslateRowGroupIds(row_groups));
  }

 private:
  inline int TranslateRowGroupId(const int row_group) const {
    return row_group_mapping_[row_group];
  }

  std::vector<int> TranslateRowGroupIds(const std::vector<int>& row_groups) const {
    std::vector<int> original_ids(row_groups.size());
    for (size_t i = 0; i < row_groups.size(); i++) {
      original_ids[i] = row_group_mapping_[row_groups[i]];
    }

    return original_ids;
  }

  std::shared_ptr<ParquetColumnIndexProvider> provider_;
  std::vector<int> row_group_mapping_;
};

bool ParquetFileFormat::Equals(const FileFormat& other) const {
  if (other.type_name() != type_name()) return false;

  const auto& other_reader_options =
      checked_cast<const ParquetFileFormat&>(other).reader_options;

  // FIXME implement comparison for decryption options
  return (reader_options.dict_columns == other_reader_options.dict_columns &&
          reader_options.coerce_int96_timestamp_unit ==
              other_reader_options.coerce_int96_timestamp_unit);
}

ParquetFileFormat::ParquetFileFormat(const parquet::ReaderProperties& reader_properties)
    : FileFormat(std::make_shared<ParquetFragmentScanOptions>()) {
  auto* default_scan_opts =
      static_cast<ParquetFragmentScanOptions*>(default_fragment_scan_options.get());
  *default_scan_opts->reader_properties = reader_properties;
}

Result<bool> ParquetFileFormat::IsSupported(const FileSource& source) const {
  auto maybe_is_supported = IsSupportedParquetFile(*this, source);
  if (!maybe_is_supported.ok()) {
    return WrapSourceError(maybe_is_supported.status(), source.path());
  }
  return maybe_is_supported;
}

Result<std::shared_ptr<Schema>> ParquetFileFormat::Inspect(
    const FileSource& source) const {
  auto scan_options = std::make_shared<ScanOptions>();
  ARROW_ASSIGN_OR_RAISE(auto reader, GetReader(source, scan_options));
  std::shared_ptr<Schema> schema;
  RETURN_NOT_OK(reader->GetSchema(&schema));
  return schema;
}

Result<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReader(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options) const {
  return GetReader(source, options, /*metadata=*/nullptr);
}

Result<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReader(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<parquet::FileMetaData>& metadata) const {
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_scan_options,
      GetFragmentScanOptions<ParquetFragmentScanOptions>(kParquetTypeName, options.get(),
                                                         default_fragment_scan_options));
  auto properties =
      MakeReaderProperties(*this, parquet_scan_options.get(), options->pool);
  ARROW_ASSIGN_OR_RAISE(auto input, source.Open());
  // `parquet::ParquetFileReader::Open` will not wrap the exception as status,
  // so using `open_parquet_file` to wrap it.
  auto open_parquet_file = [&]() -> Result<std::unique_ptr<parquet::ParquetFileReader>> {
    BEGIN_PARQUET_CATCH_EXCEPTIONS
    auto reader = parquet::ParquetFileReader::Open(std::move(input),
                                                   std::move(properties), metadata);
    return reader;
    END_PARQUET_CATCH_EXCEPTIONS
  };

  auto reader_opt = open_parquet_file();
  if (!reader_opt.ok()) {
    return WrapSourceError(reader_opt.status(), source.path());
  }
  auto reader = std::move(reader_opt).ValueOrDie();

  std::shared_ptr<parquet::FileMetaData> reader_metadata = reader->metadata();
  auto arrow_properties =
      MakeArrowReaderProperties(*this, *reader_metadata, *options, *parquet_scan_options);
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  RETURN_NOT_OK(parquet::arrow::FileReader::Make(
      options->pool, std::move(reader), std::move(arrow_properties), &arrow_reader));
  return arrow_reader;
}

Future<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReaderAsync(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options) const {
  return GetReaderAsync(source, options, nullptr);
}

Future<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReaderAsync(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<parquet::FileMetaData>& metadata) const {
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_scan_options,
      GetFragmentScanOptions<ParquetFragmentScanOptions>(kParquetTypeName, options.get(),
                                                         default_fragment_scan_options));
  auto properties =
      MakeReaderProperties(*this, parquet_scan_options.get(), options->pool);

  auto self = checked_pointer_cast<const ParquetFileFormat>(shared_from_this());

  return source.OpenAsync().Then(
      [=](const std::shared_ptr<io::RandomAccessFile>& input) mutable {
        return parquet::ParquetFileReader::OpenAsync(input, std::move(properties),
                                                     metadata)
            .Then(
                [=](const std::unique_ptr<parquet::ParquetFileReader>& reader) mutable
                -> Result<std::shared_ptr<parquet::arrow::FileReader>> {
                  auto arrow_properties = MakeArrowReaderProperties(
                      *self, *reader->metadata(), *options, *parquet_scan_options);

                  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
                  RETURN_NOT_OK(parquet::arrow::FileReader::Make(
                      options->pool,
                      // TODO(ARROW-12259): workaround since we have Future<(move-only
                      // type)> It *wouldn't* be safe to const_cast reader except that
                      // here we know there are no other waiters on the reader.
                      std::move(const_cast<std::unique_ptr<parquet::ParquetFileReader>&>(
                          reader)),
                      std::move(arrow_properties), &arrow_reader));

                  return std::move(arrow_reader);
                },
                [path = source.path()](const Status& status)
                    -> Result<std::shared_ptr<parquet::arrow::FileReader>> {
                  return WrapSourceError(status, path);
                });
      });
}

struct SlicingGenerator {
  SlicingGenerator(RecordBatchGenerator source, int64_t batch_size)
      : state(std::make_shared<State>(source, batch_size)) {}

  Future<std::shared_ptr<RecordBatch>> operator()() {
    if (state->current) {
      return state->SliceOffABatch();
    } else {
      auto state_capture = state;
      return state->source().Then(
          [state_capture](const std::shared_ptr<RecordBatch>& next) {
            if (IsIterationEnd(next)) {
              return next;
            }
            state_capture->current = next;
            return state_capture->SliceOffABatch();
          });
    }
  }

  struct State {
    State(RecordBatchGenerator source, int64_t batch_size)
        : source(std::move(source)), current(), batch_size(batch_size) {}

    std::shared_ptr<RecordBatch> SliceOffABatch() {
      if (current->num_rows() <= batch_size) {
        auto sliced = current;
        current = nullptr;
        return sliced;
      }
      auto slice = current->Slice(0, batch_size);
      current = current->Slice(batch_size);
      return slice;
    }

    RecordBatchGenerator source;
    std::shared_ptr<RecordBatch> current;
    int64_t batch_size;
  };
  std::shared_ptr<State> state;
};

Result<RecordBatchGenerator> ParquetFileFormat::ScanBatchesAsync(
    const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<FileFragment>& file) const {
  auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(file);
  std::vector<int> row_groups;
  bool pre_filtered = false;
  // If RowGroup metadata is cached completely we can pre-filter RowGroups before opening
  // a FileReader, potentially avoiding IO altogether if all RowGroups are excluded due to
  // prior statistics knowledge. In the case where a RowGroup doesn't have statistics
  // metadata, it will not be excluded.
  if (parquet_fragment->metadata() != nullptr) {
  // metdata, it will not be excluded.
  ARROW_ASSIGN_OR_RAISE(const auto has_complete_metadata,
                        parquet_fragment->HasCompleteMetadata());
  if (has_complete_metadata) {
    ARROW_ASSIGN_OR_RAISE(row_groups, parquet_fragment->FilterRowGroups(options->filter));
    pre_filtered = true;
    if (row_groups.empty()) return MakeEmptyGenerator<std::shared_ptr<RecordBatch>>();
  }
  // Open the reader and pay the real IO cost.
  auto make_generator =
      [this, options, parquet_fragment, pre_filtered,
       row_groups](const std::shared_ptr<parquet::arrow::FileReader>& reader) mutable
      -> Result<RecordBatchGenerator> {
    // Ensure that parquet_fragment has FileMetaData
    RETURN_NOT_OK(parquet_fragment->EnsureCompleteMetadata(options->filter, reader));
    if (!pre_filtered) {
      // row groups were not already filtered; do this now
      ARROW_ASSIGN_OR_RAISE(row_groups,
                            parquet_fragment->FilterRowGroups(options->filter));
      if (row_groups.empty()) return MakeEmptyGenerator<std::shared_ptr<RecordBatch>>();
    }
    ARROW_ASSIGN_OR_RAISE(auto column_projection,
                          InferColumnProjection(*reader, *options));

    // TODO: filter by column groups

    ARROW_ASSIGN_OR_RAISE(
        auto parquet_scan_options,
        GetFragmentScanOptions<ParquetFragmentScanOptions>(
            kParquetTypeName, options.get(), default_fragment_scan_options));
    int batch_readahead = options->batch_readahead;
    int64_t rows_to_readahead = batch_readahead * options->batch_size;
    ARROW_ASSIGN_OR_RAISE(auto generator,
                          reader->GetRecordBatchGenerator(
                              reader, row_groups, column_projection,
                              ::arrow::internal::GetCpuThreadPool(), rows_to_readahead));
    RecordBatchGenerator sliced =
        SlicingGenerator(std::move(generator), options->batch_size);
    if (batch_readahead == 0) {
      return sliced;
    }
    RecordBatchGenerator sliced_readahead =
        MakeSerialReadaheadGenerator(std::move(sliced), batch_readahead);
    return sliced_readahead;
  };
  auto generator = MakeFromFuture(
      GetReaderAsync(parquet_fragment->source(), options, parquet_fragment->metadata())
          .Then(std::move(make_generator)));
  WRAP_ASYNC_GENERATOR_WITH_CHILD_SPAN(
      generator, "arrow::dataset::ParquetFileFormat::ScanBatchesAsync::Next");
  return generator;
}

Future<std::optional<int64_t>> ParquetFileFormat::CountRows(
    const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
    const std::shared_ptr<ScanOptions>& options) {
  auto parquet_file = checked_pointer_cast<ParquetFileFragment>(file);
  ARROW_ASSIGN_OR_RAISE(const auto has_complete_metadata,
                        parquet_file->HasCompleteMetadata(predicate));

  if (has_complete_metadata) {
    ARROW_ASSIGN_OR_RAISE(auto maybe_count,
                          parquet_file->TryCountRows(std::move(predicate)));
    return Future<std::optional<int64_t>>::MakeFinished(maybe_count);
  } else {
    return DeferNotOk(options->io_context.executor()->Submit(
        [parquet_file, predicate]() -> Result<std::optional<int64_t>> {
          RETURN_NOT_OK(parquet_file->EnsureCompleteMetadata());
          return parquet_file->TryCountRows(predicate);
        }));
  }
}

Result<std::shared_ptr<ParquetFileFragment>> ParquetFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema, std::vector<int> row_groups,
    std::shared_ptr<ParquetColumnIndexProvider> column_index_provider) {
  return std::shared_ptr<ParquetFileFragment>(
      new ParquetFileFragment(std::move(source), shared_from_this(),
                              std::move(partition_expression), std::move(physical_schema),
                              std::move(row_groups), std::move(column_index_provider)));
}

Result<std::shared_ptr<FileFragment>> ParquetFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema) {
  return std::shared_ptr<FileFragment>(new ParquetFileFragment(
      std::move(source), shared_from_this(), std::move(partition_expression),
      std::move(physical_schema), std::nullopt));
}

//
// ParquetFileWriter, ParquetFileWriteOptions
//

std::shared_ptr<FileWriteOptions> ParquetFileFormat::DefaultWriteOptions() {
  std::shared_ptr<ParquetFileWriteOptions> options(
      new ParquetFileWriteOptions(shared_from_this()));
  options->writer_properties = parquet::default_writer_properties();
  options->arrow_writer_properties = parquet::default_arrow_writer_properties();
  return options;
}

Result<std::shared_ptr<FileWriter>> ParquetFileFormat::MakeWriter(
    std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
    std::shared_ptr<FileWriteOptions> options,
    fs::FileLocator destination_locator) const {
  if (!Equals(*options->format())) {
    return Status::TypeError("Mismatching format/write options");
  }

  auto parquet_options = checked_pointer_cast<ParquetFileWriteOptions>(options);

  std::unique_ptr<parquet::arrow::FileWriter> parquet_writer;
  ARROW_ASSIGN_OR_RAISE(parquet_writer, parquet::arrow::FileWriter::Open(
                                            *schema, default_memory_pool(), destination,
                                            parquet_options->writer_properties,
                                            parquet_options->arrow_writer_properties));

  return std::shared_ptr<FileWriter>(
      new ParquetFileWriter(std::move(destination), std::move(parquet_writer),
                            std::move(parquet_options), std::move(destination_locator)));
}

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<io::OutputStream> destination,
                                     std::shared_ptr<parquet::arrow::FileWriter> writer,
                                     std::shared_ptr<ParquetFileWriteOptions> options,
                                     fs::FileLocator destination_locator)
    : FileWriter(writer->schema(), std::move(options), std::move(destination),
                 std::move(destination_locator)),
      parquet_writer_(std::move(writer)) {}

Status ParquetFileWriter::Write(const std::shared_ptr<RecordBatch>& batch) {
  ARROW_ASSIGN_OR_RAISE(auto table, Table::FromRecordBatches(batch->schema(), {batch}));
  return parquet_writer_->WriteTable(*table, batch->num_rows());
}

Future<> ParquetFileWriter::FinishInternal() {
  return DeferNotOk(destination_locator_.filesystem->io_context().executor()->Submit(
      [this]() { return parquet_writer_->Close(); }));
}

//
// ParquetFileFragment
//

ParquetFileFragment::ParquetFileFragment(FileSource source,
                                         std::shared_ptr<FileFormat> format,
                                         compute::Expression partition_expression,
                                         std::shared_ptr<Schema> physical_schema,
                                         std::optional<std::vector<int>> row_groups)
ParquetFileFragment::ParquetFileFragment(
    FileSource source, std::shared_ptr<FileFormat> format,
    compute::Expression partition_expression, std::shared_ptr<Schema> physical_schema,
    util::optional<std::vector<int>> row_groups,
    std::shared_ptr<ParquetColumnIndexProvider> column_index_provider)
    : FileFragment(std::move(source), std::move(format), std::move(partition_expression),
                   std::move(physical_schema)),
      row_groups_(std::move(row_groups)),
      column_index_provider_(std::move(column_index_provider)) {}

const std::shared_ptr<ParquetFileFormat> ParquetFileFragment::parquet_format() const {
  return checked_pointer_cast<ParquetFileFormat>(format_);
}

Result<bool> ParquetFileFragment::HasCompleteMetadata(
    const util::optional<compute::Expression>& maybe_predicate) {
  auto lock = physical_schema_mutex_.Lock();
  if (metadata_ == nullptr) {
    return false;
  } else if (maybe_predicate.has_value() && row_groups_.has_value()) {
    DCHECK_NE(column_index_provider_, nullptr);
    return column_index_provider_->HasColumnIndexes(maybe_predicate.value(),
                                                    *row_groups_);
  }

  return true;
}

Status ParquetFileFragment::EnsureCompleteMetadata(
    const util::optional<compute::Expression>& maybe_predicate,
    std::shared_ptr<parquet::arrow::FileReader> reader) {
  {
    auto lock = physical_schema_mutex_.Lock();
    if (metadata_ != nullptr) {
      if (maybe_predicate.has_value() && row_groups_.has_value()) {
        DCHECK_NE(column_index_provider_, nullptr);
        RETURN_NOT_OK(column_index_provider_->EnsureCompleteColumnIndexes(
            maybe_predicate.value(), *row_groups_));
      }

      return Status::OK();
    }
  }

  if (reader == nullptr) {
    auto scan_options = std::make_shared<ScanOptions>();
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          parquet_format()->GetReader(source_, scan_options));
    return EnsureCompleteMetadata(maybe_predicate, std::move(reader));
  }

  auto lock = physical_schema_mutex_.Lock();
  std::shared_ptr<Schema> schema;
  RETURN_NOT_OK(reader->GetSchema(&schema));
  if (physical_schema_ && !physical_schema_->Equals(*schema)) {
    return Status::Invalid("Fragment initialized with physical schema ",
                           *physical_schema_, " but ", source_.path(), " has schema ",
                           *schema);
  }
  physical_schema_ = std::move(schema);

  if (!row_groups_) {
    row_groups_ = Iota(reader->num_row_groups());
  }

  ARROW_ASSIGN_OR_RAISE(
      auto manifest,
      GetSchemaManifest(*reader->parquet_reader()->metadata(), reader->properties()));
  RETURN_NOT_OK(SetMetadata(reader->parquet_reader()->metadata(), std::move(manifest)));

  column_index_provider_ = ParquetColumnIndexProviderImpl::Make(
      parquet_format(), source_, metadata_, physical_schema_, manifest_);
  if (maybe_predicate.has_value() && row_groups_.has_value()) {
    RETURN_NOT_OK(column_index_provider_->EnsureCompleteColumnIndexes(
        maybe_predicate.value(), *row_groups_));
  }

  return Status::OK();
}

Status ParquetFileFragment::SetMetadata(
    std::shared_ptr<parquet::FileMetaData> metadata,
    std::shared_ptr<parquet::arrow::SchemaManifest> manifest) {
  DCHECK(row_groups_.has_value());

  metadata_ = std::move(metadata);
  manifest_ = std::move(manifest);

  statistics_expressions_.resize(row_groups_->size(), compute::literal(true));
  statistics_expressions_complete_.resize(physical_schema_->num_fields(), false);

  for (int row_group : *row_groups_) {
    // Ensure RowGroups are indexing valid RowGroups before augmenting.
    if (row_group < metadata_->num_row_groups()) continue;

    return Status::IndexError("ParquetFileFragment references row group ", row_group,
                              " but ", source_.path(), " only has ",
                              metadata_->num_row_groups(), " row groups");
  }

  return Status::OK();
}

Result<FragmentVector> ParquetFileFragment::SplitByRowGroup(
    compute::Expression predicate) {
  RETURN_NOT_OK(EnsureCompleteMetadata(predicate));
  ARROW_ASSIGN_OR_RAISE(auto row_groups, FilterRowGroups(predicate));

  FragmentVector fragments(row_groups.size());
  int i = 0;
  for (int row_group : row_groups) {
    ARROW_ASSIGN_OR_RAISE(
        auto fragment,
        parquet_format()->MakeFragment(source_, partition_expression(), physical_schema_,
                                       {row_group}, column_index_provider_));

    RETURN_NOT_OK(fragment->SetMetadata(metadata_, manifest_));
    fragments[i++] = std::move(fragment);
  }

  return fragments;
}

Result<std::shared_ptr<Fragment>> ParquetFileFragment::Subset(
    compute::Expression predicate) {
  RETURN_NOT_OK(EnsureCompleteMetadata());
  ARROW_ASSIGN_OR_RAISE(auto row_groups, FilterRowGroups(predicate));
  return Subset(std::move(row_groups));
}

Result<std::shared_ptr<Fragment>> ParquetFileFragment::Subset(
    std::vector<int> row_groups) {
  RETURN_NOT_OK(EnsureCompleteMetadata());
  ARROW_ASSIGN_OR_RAISE(
      auto new_fragment,
      parquet_format()->MakeFragment(source_, partition_expression(), physical_schema_,
                                     std::move(row_groups), column_index_provider_));

  RETURN_NOT_OK(new_fragment->SetMetadata(metadata_, manifest_));
  return new_fragment;
}

inline void FoldingAnd(compute::Expression* l, compute::Expression r) {
  if (*l == compute::literal(true)) {
    *l = std::move(r);
  } else {
    *l = and_(std::move(*l), std::move(r));
  }
}

Result<std::vector<int>> ParquetFileFragment::FilterRowGroups(
    compute::Expression predicate) {
  std::vector<int> row_groups;
  ARROW_ASSIGN_OR_RAISE(auto expressions, TestRowGroups(std::move(predicate)));

  auto lock = physical_schema_mutex_.Lock();
  DCHECK(expressions.empty() || (expressions.size() == row_groups_->size()));
  for (size_t i = 0; i < expressions.size(); i++) {
    if (expressions[i].IsSatisfiable()) {
      row_groups.push_back(row_groups_->at(i));
    }
  }
  return row_groups;
}

Result<compute::Expression> ParquetFileFragment::SimplifyPredicate(
    compute::Expression predicate) {
  ARROW_ASSIGN_OR_RAISE(
      predicate, SimplifyWithGuarantee(std::move(predicate), partition_expression_));

  return predicate;
}

Result<std::vector<compute::Expression>> ParquetFileFragment::TestRowGroups(
    compute::Expression predicate) {
  ARROW_ASSIGN_OR_RAISE(const auto has_complete_metadata, HasCompleteMetadata());
  DCHECK(has_complete_metadata);

  auto lock = physical_schema_mutex_.Lock();

  ARROW_ASSIGN_OR_RAISE(predicate, SimplifyPredicate(std::move(predicate)));

  if (!predicate.IsSatisfiable()) {
    return std::vector<compute::Expression>{};
  }

  for (const FieldRef& ref : FieldsInExpression(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto match, ref.FindOneOrNone(*physical_schema_));

    if (match.empty()) continue;
    if (statistics_expressions_complete_[match[0]]) continue;
    statistics_expressions_complete_[match[0]] = true;

    const SchemaField& schema_field = manifest_->schema_fields[match[0]];

    int i = 0;
    for (int row_group : *row_groups_) {
      auto row_group_metadata = metadata_->RowGroup(row_group);

      if (auto minmax =
              ColumnChunkStatisticsAsExpression(schema_field, *row_group_metadata)) {
        FoldingAnd(&statistics_expressions_[i], std::move(*minmax));
        ARROW_ASSIGN_OR_RAISE(statistics_expressions_[i],
                              statistics_expressions_[i].Bind(*physical_schema_));
      }

      ++i;
    }
  }

  std::vector<compute::Expression> row_groups(row_groups_->size());
  for (size_t i = 0; i < row_groups_->size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto row_group_predicate,
                          SimplifyWithGuarantee(predicate, statistics_expressions_[i]));
    row_groups[i] = std::move(row_group_predicate);
  }
  return row_groups;
}

Result<std::optional<int64_t>> ParquetFileFragment::TryCountRows(
    compute::Expression predicate) {
  ARROW_ASSIGN_OR_RAISE(const auto has_complete_metadata, HasCompleteMetadata(predicate));
  DCHECK(has_complete_metadata);

  if (ExpressionHasFieldRefs(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto expressions, TestRowGroups(std::move(predicate)));
    int64_t rows = 0;
    for (size_t i = 0; i < row_groups_->size(); i++) {
      // If the row group is entirely excluded, exclude it from the row count
      if (!expressions[i].IsSatisfiable()) continue;
      // Unless the row group is entirely included, bail out of fast path
      if (expressions[i] != compute::literal(true)) return std::nullopt;
      BEGIN_PARQUET_CATCH_EXCEPTIONS
      rows += metadata()->RowGroup((*row_groups_)[i])->num_rows();
      END_PARQUET_CATCH_EXCEPTIONS
    }
    return rows;
  }
  return metadata()->num_rows();
}

//
// ParquetFragmentScanOptions
//

ParquetFragmentScanOptions::ParquetFragmentScanOptions() {
  reader_properties = std::make_shared<parquet::ReaderProperties>();
  arrow_reader_properties =
      std::make_shared<parquet::ArrowReaderProperties>(/*use_threads=*/false);
}

//
// ParquetDatasetFactory
//

static inline Result<std::string> FileFromRowGroup(
    fs::FileSystem* filesystem, const std::string& base_path,
    const parquet::RowGroupMetaData& row_group, bool validate_column_chunk_paths) {
  constexpr auto prefix = "Extracting file path from RowGroup failed. ";

  if (row_group.num_columns() == 0) {
    return Status::Invalid(prefix,
                           "RowGroup must have a least one column to extract path.");
  }

  auto path = row_group.ColumnChunk(0)->file_path();
  if (path == "") {
    return Status::Invalid(
        prefix,
        "The column chunks' file paths should be set, but got an empty file path.");
  }

  if (validate_column_chunk_paths) {
    for (int i = 1; i < row_group.num_columns(); ++i) {
      const auto& column_path = row_group.ColumnChunk(i)->file_path();
      if (column_path != path) {
        return Status::Invalid(prefix, "Path '", column_path, "' not equal to path '",
                               path, ", for ColumnChunk at index ", i,
                               "; ColumnChunks in a RowGroup must have the same path.");
      }
    }
  }

  path = fs::internal::JoinAbstractPath(
      std::vector<std::string>{base_path, std::move(path)});
  // Normalizing path is required for Windows.
  return filesystem->NormalizePath(std::move(path));
}

Result<std::shared_ptr<Schema>> GetSchema(
    const parquet::FileMetaData& metadata,
    const parquet::ArrowReaderProperties& properties) {
  std::shared_ptr<Schema> schema;
  RETURN_NOT_OK(parquet::arrow::FromParquetSchema(
      metadata.schema(), properties, metadata.key_value_metadata(), &schema));
  return schema;
}

Result<std::shared_ptr<DatasetFactory>> ParquetDatasetFactory::Make(
    const std::string& metadata_path, std::shared_ptr<fs::FileSystem> filesystem,
    std::shared_ptr<ParquetFileFormat> format, ParquetFactoryOptions options) {
  // Paths in ColumnChunk are relative to the `_metadata` file. Thus, the base
  // directory of all parquet files is `dirname(metadata_path)`.
  auto dirname = arrow::fs::internal::GetAbstractPathParent(metadata_path).first;
  return Make({metadata_path, filesystem}, dirname, filesystem, std::move(format),
              std::move(options));
}

Result<std::shared_ptr<DatasetFactory>> ParquetDatasetFactory::Make(
    const FileSource& metadata_source, const std::string& base_path,
    std::shared_ptr<fs::FileSystem> filesystem, std::shared_ptr<ParquetFileFormat> format,
    ParquetFactoryOptions options) {
  DCHECK_NE(filesystem, nullptr);
  DCHECK_NE(format, nullptr);

  // By automatically setting the options base_dir to the metadata's base_path,
  // we provide a better experience for user providing Partitioning that are
  // relative to the base_dir instead of the full path.
  if (options.partition_base_dir.empty()) {
    options.partition_base_dir = base_path;
  }

  auto scan_options = std::make_shared<ScanOptions>();
  ARROW_ASSIGN_OR_RAISE(auto reader, format->GetReader(metadata_source, scan_options));
  std::shared_ptr<parquet::FileMetaData> metadata = reader->parquet_reader()->metadata();

  if (metadata->num_columns() == 0) {
    return Status::Invalid(
        "ParquetDatasetFactory must contain a schema with at least one column");
  }

  auto properties = MakeArrowReaderProperties(*format, *metadata);
  ARROW_ASSIGN_OR_RAISE(auto physical_schema, GetSchema(*metadata, properties));
  ARROW_ASSIGN_OR_RAISE(auto manifest, GetSchemaManifest(*metadata, properties));

  std::vector<std::pair<std::string, std::vector<int>>> paths_with_row_group_ids;
  std::unordered_map<std::string, int> paths_to_index;

  for (int i = 0; i < metadata->num_row_groups(); i++) {
    auto row_group = metadata->RowGroup(i);
    ARROW_ASSIGN_OR_RAISE(auto path,
                          FileFromRowGroup(filesystem.get(), base_path, *row_group,
                                           options.validate_column_chunk_paths));

    // Insert the path, or increase the count of row groups. It will be assumed that the
    // RowGroup of a file are ordered exactly as in the metadata file.
    auto inserted_index = paths_to_index.emplace(
        std::move(path), static_cast<int>(paths_with_row_group_ids.size()));
    if (inserted_index.second) {
      paths_with_row_group_ids.push_back({inserted_index.first->first, {}});
    }
    paths_with_row_group_ids[inserted_index.first->second].second.push_back(i);
  }

  auto column_index_provider = ParquetColumnIndexProviderImpl::Make(
      format, metadata_source, metadata, physical_schema, manifest);

  return std::shared_ptr<DatasetFactory>(new ParquetDatasetFactory(
      std::move(filesystem), std::move(format), std::move(metadata), std::move(manifest),
      std::move(physical_schema), base_path, std::move(options),
      std::move(paths_with_row_group_ids), std::move(column_index_provider)));
}

Result<std::vector<std::shared_ptr<FileFragment>>>
ParquetDatasetFactory::CollectParquetFragments(const Partitioning& partitioning) {
  std::vector<std::shared_ptr<FileFragment>> fragments(paths_with_row_group_ids_.size());

  size_t i = 0;
  for (const auto& e : paths_with_row_group_ids_) {
    const auto& path = e.first;
    const auto& original_row_group_ids = e.second;
    auto metadata_subset = metadata_->Subset(original_row_group_ids);

    auto row_groups = Iota(metadata_subset->num_row_groups());

    auto partition_expression =
        partitioning.Parse(StripPrefix(path, options_.partition_base_dir))
            .ValueOr(compute::literal(true));

    ARROW_ASSIGN_OR_RAISE(
        auto fragment,
        format_->MakeFragment({path, filesystem_}, std::move(partition_expression),
                              physical_schema_, std::move(row_groups),
                              ParquetColumnIndexProviderView::Make(
                                  column_index_provider_, original_row_group_ids)));

    RETURN_NOT_OK(fragment->SetMetadata(metadata_subset, manifest_));
    fragments[i++] = std::move(fragment);
  }

  return fragments;
}

Result<std::vector<std::shared_ptr<Schema>>> ParquetDatasetFactory::InspectSchemas(
    InspectOptions options) {
  // The physical_schema from the _metadata file is always yielded
  std::vector<std::shared_ptr<Schema>> schemas = {physical_schema_};

  if (auto factory = options_.partitioning.factory()) {
    // Gather paths found in RowGroups' ColumnChunks.
    std::vector<std::string> stripped(paths_with_row_group_ids_.size());

    size_t i = 0;
    for (const auto& e : paths_with_row_group_ids_) {
      stripped[i++] = StripPrefixAndFilename(e.first, options_.partition_base_dir);
    }
    ARROW_ASSIGN_OR_RAISE(auto partition_schema, factory->Inspect(stripped));

    schemas.push_back(std::move(partition_schema));
  } else {
    schemas.push_back(options_.partitioning.partitioning()->schema());
  }

  return schemas;
}

Result<std::shared_ptr<Dataset>> ParquetDatasetFactory::Finish(FinishOptions options) {
  std::shared_ptr<Schema> schema = options.schema;
  bool schema_missing = schema == nullptr;
  if (schema_missing) {
    ARROW_ASSIGN_OR_RAISE(schema, Inspect(options.inspect_options));
  }

  std::shared_ptr<Partitioning> partitioning = options_.partitioning.partitioning();
  if (partitioning == nullptr) {
    auto factory = options_.partitioning.factory();
    ARROW_ASSIGN_OR_RAISE(partitioning, factory->Finish(schema));
  }

  ARROW_ASSIGN_OR_RAISE(auto fragments, CollectParquetFragments(*partitioning));
  return FileSystemDataset::Make(std::move(schema), compute::literal(true), format_,
                                 filesystem_, std::move(fragments),
                                 std::move(partitioning));
}

}  // namespace dataset
}  // namespace arrow
