#include "arrow/util/row_ranges.h"

#include <limits>

#include "arrow/util/logging.h"

namespace arrow {
namespace util {

void TrySetTailEnd(std::vector<RowRange>& ranges, const int64_t to) {
  while (!ranges.empty()) {
    if (ranges.back().from >= to) {
      ranges.pop_back();
    } else {
      ranges.back().to = to;
      break;
    }
  }
}

void AddTail(std::vector<RowRange>& ranges, const RowRange& incoming) {
  if (incoming.empty()) {
    return;
  } else if (ranges.empty()) {
    ranges.push_back(incoming);
    return;
  }

  const auto& tail = ranges.back();
  DCHECK_LE(tail.from, incoming.from);

  if (tail.to < incoming.from) {
    // Disjunct case, add incoming range as is.
    // tail: partial?[0, 20), incoming: partial?[22, 25)
    ranges.push_back(incoming);
    // result[-2:]: partial?[0, 20), partial?[22, 25)
  } else if (tail.to < incoming.to) {
    // Half intersect.
    // tail: partial?[0, 20) ; incoming: partial?[12, 25)
    if (tail.partial == incoming.partial) {
      // Both tail and incoming are of the same kind.
      // Tail should eat incoming,
      TrySetTailEnd(ranges, incoming.to);
      // result[-1]: partial?[0, 25)
    } else if (incoming.partial) {
      // Tail must be non partial. Add non intersected part of incoming as partial
      ranges.push_back({true, tail.to, incoming.to});
      // result[-1]: partial[20, 25)
    } else {
      // Tail must be partial. Trim it to incoming.from
      TrySetTailEnd(ranges, incoming.from);
      ranges.push_back(incoming);
      // result[-2:]: partial[0, 12), [12, 25)
    }
  } else if (tail.partial && !incoming.partial) {
    // Tail covers whole of the incoming range.
    // Split the tail into 2 and insert the incoming in between.
    // tail: partial[7, 20), incoming: [12, 15)
    auto old_to = tail.to;
    TrySetTailEnd(ranges, incoming.from);
    ranges.push_back(incoming);
    if (old_to > incoming.to) {  // Add remaining if any
      ranges.push_back({true, incoming.to, old_to});
    }

    // result[-3:]: partial[7, 12), [12, 15), partial[15, 20)
  }
}

void RowRanges::Add(const RowRange& range) {
  if (range.empty()) {
    return;
  }

  // binary search insert position
  auto it = std::upper_bound(
      ranges_.begin(), ranges_.end(), range,
      [](const RowRange& a, const RowRange& b) { return a.from < b.from; });
  if (it == ranges_.end()) {
    // Optimized case.
    AddTail(ranges_, range);
  } else {
    std::vector<RowRange> new_ranges;
    // Move earlier ranges as is.
    std::copy(ranges_.begin(), it, std::back_inserter(new_ranges));

    // Add new range as tail.
    AddTail(new_ranges, range);

    // Add rest of ranges as tail.
    while (it != ranges_.end()) {
      AddTail(new_ranges, *it++);
    }

    // Set new ranges as our ranges.
    ranges_ = std::move(new_ranges);
  }
}

RowRanges RowRanges::Union(const RowRanges& other) const {
  std::vector<RowRange> result_ranges;
  auto i = ranges_.begin();
  auto j = other.ranges().begin();
  while (i != ranges_.end() || j != other.ranges().end()) {
    if (i == ranges_.end() || (j != other.ranges().end() && j->from < i->from)) {
      AddTail(result_ranges, *(j++));
    } else {
      AddTail(result_ranges, *(i++));
    }
  }

  return RowRanges(result_ranges);
}

RowRanges RowRanges::Intersect(const RowRanges& other) const {
  std::vector<RowRange> result_ranges;

  RowRange residual{false, std::numeric_limits<int64_t>::min(),
                    std::numeric_limits<int64_t>::min()};
  auto i = ranges_.begin();
  auto j = other.ranges_.begin();
  const RowRange* incoming = nullptr;
  while (i != ranges_.end() || j != other.ranges_.end()) {
    if (i == ranges_.end() || (j != other.ranges_.end() && j->from < i->from)) {
      incoming = &*(j++);
    } else {
      incoming = &*(i++);
    }

    if (residual.to < incoming->from) {
      // Incoming is disjunct. Discard existing residiual and let incoming stay.
      residual = *incoming;
    } else if (residual.to > incoming->to) {
      // Residual completely covers incoming.
      // Add intersection to result
      AddTail(result_ranges,
              {residual.partial || incoming->partial, incoming->from, incoming->to});
      // Trim residuals head to incoming end.
      residual.from = incoming->to;
    } else {
      // Incoming is not covered by residual. Imcoming must survive.
      // Add intersection to result
      AddTail(result_ranges,
              {residual.partial || incoming->partial, incoming->from, residual.to});

      const auto old_to = residual.to;
      // Incoming is the new residual.
      residual = *incoming;
      // Trim new residual's head to old end.
      residual.from = old_to;
    }
  }

  return RowRanges(std::move(result_ranges));
}

RowRanges RowRanges::Negate() const {
  std::vector<RowRange> result_ranges;
  int64_t last = std::numeric_limits<int64_t>::min();
  for (const auto& range : ranges_) {
    if (range.from > last) {
      AddTail(result_ranges, {false, last, range.from});
    }

    if (range.partial) {
      AddTail(result_ranges, range);
    }

    last = range.to;
  }

  if (last < std::numeric_limits<int64_t>::max()) {
    AddTail(result_ranges, {false, last, std::numeric_limits<int64_t>::max()});
  }

  return RowRanges(result_ranges);
}

}  // namespace util
}  // namespace arrow