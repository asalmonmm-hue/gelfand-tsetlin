#ifndef GT_COEFFICIENT_H
#define GT_COEFFICIENT_H

#include <stddef.h>
#include <stdint.h>

enum gt_coefficient_status {
    GT_COEFFICIENT_OK,
    GT_COEFFICIENT_INVALID,
    GT_COEFFICIENT_OVERFLOW
};

/*
 * Coefficient of the target basis vector in E_{i,j} applied to the source.
 * Indices are one-based. Both borrowed buffers contain valid rank-N patterns
 * in top-first row order, with entries normalized by the same origin to
 * nonnegative int64_t values. For the same representation, use the last
 * entry of the top row as the origin.
 *
 * On success the outputs form a reduced rational with positive denominator.
 * A diagonal coefficient at an unchanged target omits the origin, which the
 * caller can add back exactly. Unsupported targets return 0/1.
 * Overflow reports an out-of-range intermediate or final rational. The
 * output pointers must be distinct. The function uses constant stack space.
 */
enum gt_coefficient_status
gt_coefficient(size_t rank, size_t i, size_t j,
               const int64_t *source, const int64_t *target,
               int64_t *numerator, int64_t *denominator);

#endif
