#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/util/row_ranges.h"

namespace arrow {
namespace util {

using RangeVector = std::vector<RowRange>;

std::ostream& operator<<(std::ostream& os, const RowRange& r) {
  os << (r.partial ? "partial" : "") << "[" << r.from << ", " << r.to << ")";
  return os;
}

TEST(RowRangeTests, Add) {
  RowRanges ranges;

  EXPECT_TRUE(ranges.empty());

  // Add empty range into empty ranges does nothing
  ranges.Add({false, 3, 3});
  EXPECT_TRUE(ranges.empty());

  // Add into the empty ranges
  ranges.Add({false, 5, 7});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}}));

  // Add empty into the head does nothing
  ranges.Add({false, 2, 2});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}}));

  // Add empty into middle does nothing
  ranges.Add({false, 6, 6});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}}));

  // Add empty into the tail does nothing
  ranges.Add({false, 9, 9});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}}));

  // Add Disjunct to tail.
  ranges.Add({false, 14, 17});
  EXPECT_EQ(ranges.ranges().size(), 2);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}, {false, 14, 17}}));

  // Add Touching to tail same kind merges tail.
  ranges.Add({false, 17, 23});
  EXPECT_EQ(ranges.ranges().size(), 2);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}, {false, 14, 23}}));

  // Add Intersects Tail same kind merges tail.
  ranges.Add({false, 19, 27});
  EXPECT_EQ(ranges.ranges().size(), 2);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 7}, {false, 14, 27}}));

  // Add Disjunct Middle.
  ranges.Add({false, 10, 12});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 7}, {false, 10, 12}, {false, 14, 27}}));

  // Add Touching Middle merges.
  ranges.Add({false, 7, 8});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 8}, {false, 10, 12}, {false, 14, 27}}));

  // Add Intersects with single in the middle merges.
  ranges.Add({false, 11, 13});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 8}, {false, 10, 13}, {false, 14, 27}}));

  // Add Intersects with multiple in the middle merges.
  ranges.Add({false, 7, 15});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 27}}));

  // Add Touching partial does not merge
  ranges.Add({true, 27, 33});
  EXPECT_EQ(ranges.ranges().size(), 2);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 27}, {true, 27, 33}}));

  // Add Touching non partial eats partial right side
  ranges.Add({false, 31, 37});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 27}, {true, 27, 31}, {false, 31, 37}}));

  // Add Touching non partial eats partial left side
  ranges.Add({false, 25, 29});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 29}, {true, 29, 31}, {false, 31, 37}}));

  // Non partial eats merges everything
  ranges.Add({false, 22, 35});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 37}}));

  // Partial intersection has no effect
  ranges.Add({true, 15, 19});
  EXPECT_EQ(ranges.ranges().size(), 1);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 37}}));

  // Misc
  ranges.Add({false, 40, 49});
  ranges.Add({false, 55, 59});
  EXPECT_EQ(ranges.ranges().size(), 3);
  EXPECT_EQ(ranges.ranges(),
            RangeVector({{false, 5, 37}, {false, 40, 49}, {false, 55, 59}}));

  // Partial spanning multiple ranges fills in the blankc
  ranges.Add({true, 25, 57});
  EXPECT_EQ(ranges.ranges().size(), 5);
  EXPECT_EQ(ranges.ranges(), RangeVector({{false, 5, 37},
                                          {true, 37, 40},
                                          {false, 40, 49},
                                          {true, 49, 55},
                                          {false, 55, 59}}));
}

TEST(RowRangeTests, Union_Simple) {
  RowRanges a_non_partial{{false, 7, 15}};
  RowRanges a_partial{{true, 7, 15}};
  RowRanges b_non_partial{{false, 3, 9}};
  RowRanges b_partial{{true, 3, 9}};

  // Self union
  EXPECT_EQ(a_non_partial.Union(a_non_partial).ranges(), a_non_partial.ranges());
  EXPECT_EQ(a_partial.Union(a_partial).ranges(), a_partial.ranges());

  // Non Partial Unions
  EXPECT_EQ(a_non_partial.Union(b_non_partial).ranges(), RangeVector({{false, 3, 15}}));

  // Partial Unions
  EXPECT_EQ(a_partial.Union(b_partial).ranges(), RangeVector({{true, 3, 15}}));

  // Non partial triumps
  EXPECT_EQ(a_partial.Union(a_non_partial).ranges(), a_non_partial.ranges());
  EXPECT_EQ(a_non_partial.Union(a_partial).ranges(), a_non_partial.ranges());

  // Mixed
  EXPECT_EQ(a_partial.Union(b_non_partial).ranges(),
            RangeVector({{false, 3, 9}, {true, 9, 15}}));
  EXPECT_EQ(a_non_partial.Union(b_partial).ranges(),
            RangeVector({{true, 3, 7}, {false, 7, 15}}));
}

RowRanges ranges_a({
    {true, 4, 5},
    {false, 9, 15},
    {true, 20, 25},
    {false, 25, 30},
});

RowRanges ranges_b({
    {false, 3, 7},
    {true, 22, 35},
});

TEST(RowRangeTests, Union_Complex) {
  // Disjoint
  EXPECT_EQ(RowRanges({false, 3, 5}).Union(RowRanges({true, 7, 9})).ranges(),
            RangeVector({{false, 3, 5}, {true, 7, 9}}));
  EXPECT_EQ(RowRanges({false, 7, 9}).Union(RowRanges({true, 3, 5})).ranges(),
            RangeVector({{true, 3, 5}, {false, 7, 9}}));

  // Intersect same kind

  // partial intersection
  EXPECT_EQ(RowRanges({false, 3, 8}).Union(RowRanges({false, 7, 9})).ranges(),
            RangeVector({{false, 3, 9}}));
  EXPECT_EQ(RowRanges({true, 3, 8}).Union(RowRanges({true, 7, 9})).ranges(),
            RangeVector({{true, 3, 9}}));
  EXPECT_EQ(RowRanges({false, 7, 9}).Union(RowRanges({false, 3, 8})).ranges(),
            RangeVector({{false, 3, 9}}));
  EXPECT_EQ(RowRanges({true, 7, 9}).Union(RowRanges({true, 3, 8})).ranges(),
            RangeVector({{true, 3, 9}}));

  // One totally covers the other
  EXPECT_EQ(RowRanges({false, 3, 8}).Union(RowRanges({false, 4, 6})).ranges(),
            RangeVector({{false, 3, 8}}));
  EXPECT_EQ(RowRanges({true, 3, 8}).Union(RowRanges({true, 4, 6})).ranges(),
            RangeVector({{true, 3, 8}}));
  EXPECT_EQ(RowRanges({false, 4, 6}).Union(RowRanges({false, 3, 8})).ranges(),
            RangeVector({{false, 3, 8}}));
  EXPECT_EQ(RowRanges({true, 4, 6}).Union(RowRanges({true, 3, 8})).ranges(),
            RangeVector({{true, 3, 8}}));

  std::vector<RowRange> ranges_list_union{
      {false, 3, 7}, {false, 9, 15}, {true, 20, 25}, {false, 25, 30}, {true, 30, 35},
  };

  EXPECT_EQ(ranges_a.Union(ranges_b).ranges(), ranges_list_union);
}

TEST(RowRangeTests, Intersect_Simple) {
  RowRanges a_non_partial{{false, 7, 15}};
  RowRanges a_partial{{true, 7, 15}};
  RowRanges b_non_partial{{false, 3, 9}};
  RowRanges b_partial{{true, 3, 9}};

  // Self intersect
  EXPECT_EQ(a_non_partial.Intersect(a_non_partial).ranges(), a_non_partial.ranges());
  EXPECT_EQ(a_partial.Intersect(a_partial).ranges(), a_partial.ranges());

  // Non Partial Intersects
  EXPECT_EQ(a_non_partial.Intersect(b_non_partial).ranges(),
            RangeVector({{false, 7, 9}}));

  // Partial Intersects
  EXPECT_EQ(a_partial.Intersect(b_partial).ranges(), RangeVector({{true, 7, 9}}));

  // Partial triumps
  EXPECT_EQ(a_partial.Intersect(a_non_partial).ranges(), a_partial.ranges());
  EXPECT_EQ(a_non_partial.Intersect(a_partial).ranges(), a_partial.ranges());

  // Mixed
  EXPECT_EQ(a_partial.Intersect(b_non_partial).ranges(), RangeVector({{true, 7, 9}}));
  EXPECT_EQ(a_non_partial.Intersect(b_partial).ranges(), RangeVector({{true, 7, 9}}));
}

TEST(RowRangeTests, Intersect_Complex) {
  std::vector<RowRange> ranges{{true, 4, 5}, {true, 22, 30}};

  EXPECT_EQ(ranges_a.Intersect(ranges_b).ranges(), ranges);
}

TEST(RowRangeTests, Invert) {
  RowRanges non_partial{{false, 7, 15}};
  RowRanges partial{{true, 7, 15}};
  // Non partial does not survive Invert
  EXPECT_EQ(non_partial.Invert().ranges(),
            RangeVector({{false, std::numeric_limits<int64_t>::min(), 7},
                         {false, 15, std::numeric_limits<int64_t>::max()}}));

  // Partial survives Invert
  EXPECT_EQ(partial.Invert().ranges(),
            RangeVector({{false, std::numeric_limits<int64_t>::min(), 7},
                         {true, 7, 15},
                         {false, 15, std::numeric_limits<int64_t>::max()}}));

  // Gap turns non partial.
  RowRanges gap({{false, 7, 15}, {false, 20, 25}});
  EXPECT_EQ(gap.Invert().ranges(),
            RangeVector({{false, std::numeric_limits<int64_t>::min(), 7},
                         {false, 15, 20},
                         {false, 25, std::numeric_limits<int64_t>::max()}}));

  RowRanges complex(
      {{true, 7, 15}, {false, 15, 25}, {true, 25, 30}, {true, 35, 40}, {false, 45, 50}});
  EXPECT_EQ(complex.Invert().ranges(),
            RangeVector({{false, std::numeric_limits<int64_t>::min(), 7},
                         {true, 7, 15},
                         {true, 25, 30},
                         {false, 30, 35},
                         {true, 35, 40},
                         {false, 40, 45},
                         {false, 50, std::numeric_limits<int64_t>::max()}}));
}

}  // namespace util
}  // namespace arrow