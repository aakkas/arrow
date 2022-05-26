#pragma once

#include <vector>

namespace arrow {
namespace util {

struct RowRange {
  bool partial;
  int64_t from;
  int64_t to;

  inline bool operator==(const RowRange& other) const noexcept {
    return partial == other.partial && from == other.from && to == other.to;
  }

  inline bool empty() const noexcept { return from >= to; }
};

class RowRanges {
 public:
  RowRanges() {}
  RowRanges(const int64_t row_count, const bool partial)
      : ranges_({{partial, 0, row_count}}) {}
  RowRanges(RowRange range) : ranges_({std::move(range)}) {}
  RowRanges(std::vector<RowRange> ranges) : ranges_(std::move(ranges)) {}
  RowRanges(const RowRanges& other) : ranges_(other.ranges()) {}

  RowRanges Negate() const;
  RowRanges Union(const RowRanges& other) const;
  RowRanges Intersect(const RowRanges& other) const;

  void Add(const RowRange& range);

  inline bool empty() const noexcept { return ranges_.empty(); }
  inline const std::vector<RowRange>& ranges() const noexcept { return ranges_; };

 private:
  std::vector<RowRange> ranges_;
};

}  // namespace util
}  // namespace arrow