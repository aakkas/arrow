#pragma once

#include <limits>
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

  static const RowRange& ALL() {
    static RowRange all{false, std::numeric_limits<int64_t>::min(),
                        std::numeric_limits<int64_t>::max()};
    return all;
  }

  static const RowRange& ALL_PARTIAL() {
    static RowRange all_partial{true, std::numeric_limits<int64_t>::min(),
                                std::numeric_limits<int64_t>::max()};
    return all_partial;
  }
};

class RowRanges {
 public:
  RowRanges() {}
  RowRanges(RowRange range) : ranges_({std::move(range)}) {}
  RowRanges(std::vector<RowRange> ranges) : ranges_(std::move(ranges)) {}
  RowRanges(const RowRanges& other) : ranges_(other.ranges()) {}

  RowRanges Invert() const;
  RowRanges Union(const RowRanges& other) const;
  RowRanges Intersect(const RowRanges& other) const;

  void Add(const RowRange& range);

  inline bool empty() const noexcept { return ranges_.empty(); }
  inline const std::vector<RowRange>& ranges() const noexcept { return ranges_; };

  static const RowRanges& NONE() {
    static RowRanges none;
    return none;
  }

  static const RowRanges& ALL() {
    static RowRanges all{RowRange::ALL()};
    return all;
  }

  static const RowRanges& ALL_PARTIAL() {
    static RowRanges all_partial{RowRange::ALL_PARTIAL()};
    return all_partial;
  }

 private:
  std::vector<RowRange> ranges_;
};

}  // namespace util
}  // namespace arrow