#ifndef GT_INNER_PRODUCT_H
#define GT_INNER_PRODUCT_H

#include "gt_coefficient.h"

/*
 * Both arrays contain validated interlacing patterns of the same highest
 * weight in contiguous, top-first rows of lengths rank, ..., 1. Entries
 * are nonnegative offsets from their common origin. The output is 0 for
 * distinct patterns and the squared norm in Molev's basis otherwise.
 * The denominator is always 1. Outputs are written only on success.
 */
enum gt_coefficient_status gt_inner_product(
    size_t rank, const int64_t *left, const int64_t *right,
    int64_t *numerator, int64_t *denominator);

#endif
