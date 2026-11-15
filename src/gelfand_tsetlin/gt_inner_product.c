#include "gt_inner_product.h"

#include <stdbool.h>

/* Multiply by (argument + gap)! / argument!, stopping on overflow. */
static bool multiply_factorial_ratio(uint64_t argument, uint64_t gap,
                                     uint64_t *product)
{
    for (uint64_t step = 0; step < gap; ++step) {
        if (argument >= (uint64_t)INT64_MAX / *product) {
            return false;
        }
        ++argument;
        *product *= argument;
    }
    return true;
}

enum gt_coefficient_status gt_inner_product(
    size_t rank, const int64_t *left, const int64_t *right,
    int64_t *numerator, int64_t *denominator)
{
    if (rank == 0 || left == NULL || right == NULL ||
        numerator == NULL || denominator == NULL) {
        return GT_COEFFICIENT_INVALID;
    }

    /* Proposition 2.4 states that the pattern basis is orthogonal. */
    if (left != right) {
        const int64_t *left_row = left;
        const int64_t *right_row = right;
        for (size_t width = rank; width > 0; --width) {
            for (size_t column = 0; column < width; ++column) {
                if (left_row[column] != right_row[column]) {
                    *numerator = 0;
                    *denominator = 1;
                    return GT_COEFFICIENT_OK;
                }
            }
            left_row += width;
            right_row += width;
        }
    }

    uint64_t product = 1;
    const int64_t *upper = left;
    for (size_t width = rank; width > 1; --width) {
        const int64_t *lower = upper + width;
        for (size_t i = 0; i < width - 1; ++i) {
            /* In both ratios the numerator argument exceeds the
               denominator argument by this same interlacing gap. */
            uint64_t gap = (uint64_t)(upper[i] - lower[i]);
            if (gap == 0) {
                continue;
            }
            for (size_t j = i; j < width - 1; ++j) {
                /* The second product uses column j + 1 in the upper row. */
                uint64_t offset = (uint64_t)(j - i);
                uint64_t within_lower = (uint64_t)(lower[i] - lower[j]) + offset;
                uint64_t across_rows = (uint64_t)(lower[i] - upper[j + 1]) + offset;
                if (!multiply_factorial_ratio(within_lower, gap, &product) ||
                    !multiply_factorial_ratio(across_rows, gap, &product)) {
                    return GT_COEFFICIENT_OVERFLOW;
                }
            }
        }
        upper = lower;
    }

    *numerator = (int64_t)product;
    *denominator = 1;
    return GT_COEFFICIENT_OK;
}
