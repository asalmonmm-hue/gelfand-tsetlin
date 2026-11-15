#include "gt_coefficient.h"

static int
triangular_size(size_t rank, size_t *size)
{
    size_t half = rank / 2;
    size_t left = rank % 2 == 0 ? half : rank;
    size_t right = rank % 2 == 0 ? rank + 1 : half + 1;

    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *size = left * right;
    return 1;
}

static uint64_t
gcd(uint64_t a, uint64_t b)
{
    while (b != 0) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static void
cancel(uint64_t *a, uint64_t *b)
{
    uint64_t divisor = gcd(*a, *b);
    *a /= divisor;
    *b /= divisor;
}

/* Every factor is nonzero. Pairing factors permits cancellation before
 * either multiplication and keeps the partial ratio reduced. */
static int
multiply_ratio(uint64_t *numerator, uint64_t *denominator,
               uint64_t above, uint64_t below)
{
    cancel(&above, &below);
    cancel(numerator, &below);
    cancel(&above, denominator);
    if (*numerator > UINT64_MAX / above || *denominator > UINT64_MAX / below) {
        return 0;
    }
    *numerator *= above;
    *denominator *= below;
    return 1;
}

/* The magnitude of a signed gap can exceed INT64_MAX. The unsigned
 * subtraction is exact even when the operands have opposite signs. */
static uint64_t
factor(int64_t a, int64_t b, int *negative)
{
    if (a < b) {
        *negative = !*negative;
        return (uint64_t)b - (uint64_t)a;
    }
    return (uint64_t)a - (uint64_t)b;
}

static enum gt_coefficient_status
adjacent_coefficient(const int64_t *row, size_t length, size_t column,
                     int step, int64_t *numerator, int64_t *denominator)
{
    size_t neighbor_length = step == 1 ? length + 1 : length - 1;
    const int64_t *neighbor = step == 1 ? row - neighbor_length : row + length;
    int64_t pivot = row[column] - (int64_t)column;
    int negative = step == 1;
    uint64_t reduced_num = 1;
    uint64_t reduced_den = 1;

    /* An exact zero takes precedence over overflow in a different factor. */
    for (size_t index = 0; index < neighbor_length; index++) {
        if (pivot == neighbor[index] - (int64_t)index) {
            return GT_COEFFICIENT_OK;
        }
    }
    for (size_t index = 0; index < neighbor_length; index++) {
        uint64_t above = factor(pivot, neighbor[index] - (int64_t)index, &negative);
        uint64_t below = 1;
        if (index < length - 1) {
            size_t other = index < column ? index : index + 1;
            below = factor(pivot, row[other] - (int64_t)other, &negative);
            if (below == 0) {
                return GT_COEFFICIENT_INVALID;
            }
        }
        if (!multiply_ratio(&reduced_num, &reduced_den, above, below)) {
            return GT_COEFFICIENT_OVERFLOW;
        }
    }

    uint64_t limit = (uint64_t)INT64_MAX + (uint64_t)negative;
    if (reduced_num > limit || reduced_den > (uint64_t)INT64_MAX) {
        return GT_COEFFICIENT_OVERFLOW;
    }
    if (negative && reduced_num == limit) {
        *numerator = INT64_MIN;
    } else {
        *numerator = negative ? -(int64_t)reduced_num : (int64_t)reduced_num;
    }
    *denominator = (int64_t)reduced_den;
    return GT_COEFFICIENT_OK;
}

enum gt_coefficient_status
gt_adjacent_coefficient(size_t rank, size_t i, size_t j,
                        const int64_t *source, size_t column,
                        int64_t *numerator, int64_t *denominator)
{
    size_t total;
    size_t tail;
    size_t length = i < j ? i : j;
    size_t distance = i < j ? j - i : i - j;
    if (source == NULL || numerator == NULL || denominator == NULL ||
        numerator == denominator || length == 0 || i > rank || j > rank ||
        distance != 1 || column >= length || rank > (uint64_t)INT64_MAX ||
        !triangular_size(rank, &total) || total > SIZE_MAX / sizeof(*source) ||
        !triangular_size(length, &tail)) {
        return GT_COEFFICIENT_INVALID;
    }

    *numerator = 0;
    *denominator = 1;
    const int64_t *row = source + total - tail;
    const int64_t *upper = row - (length + 1);
    const int64_t *lower = row + length;
    int64_t entry = row[column];
    int step = i < j ? 1 : -1;
    if ((step == 1 && entry == INT64_MAX) || (step == -1 && entry == 0)) {
        return GT_COEFFICIENT_OK;
    }
    entry += step;
    if (entry > upper[column] || entry < upper[column + 1] ||
        (column > 0 && entry > lower[column - 1]) ||
        (column + 1 < length && entry < lower[column])) {
        return GT_COEFFICIENT_OK;
    }
    return adjacent_coefficient(row, length, column, step, numerator, denominator);
}

enum gt_coefficient_status
gt_coefficient(size_t rank, size_t i, size_t j,
               const int64_t *source, const int64_t *target,
               int64_t *numerator, int64_t *denominator)
{
    size_t total;
    size_t tail;
    size_t length = i < j ? i : j;
    size_t distance = i < j ? j - i : i - j;
    if (source == NULL || target == NULL || numerator == NULL || denominator == NULL ||
        numerator == denominator || i == 0 || j == 0 || i > rank || j > rank ||
        distance > 1 || rank > (uint64_t)INT64_MAX ||
        !triangular_size(rank, &total) || total > SIZE_MAX / sizeof(*source) ||
        !triangular_size(length, &tail)) {
        return GT_COEFFICIENT_INVALID;
    }

    /* The triangle of the shorter rows is a suffix of each buffer. */
    size_t start = total - tail;
    size_t stop = start + length;
    size_t changed = total;
    int step = i < j ? 1 : -1;
    *numerator = 0;
    *denominator = 1;
    for (size_t index = 0; index < total; index++) {
        if (target[index] == source[index]) {
            continue;
        }
        if (distance == 0 || index < start || index >= stop ||
            changed != total || target[index] - source[index] != step) {
            return GT_COEFFICIENT_OK;
        }
        changed = index;
    }

    const int64_t *row = source + start;
    if (distance == 0) {
        const int64_t *lower = row + length;
        int64_t weight = row[length - 1];
        /* Interlacing bounds the sum by row[0], avoiding overflowing row sums. */
        for (size_t index = 0; index < length - 1; index++) {
            int64_t gap = row[index] - lower[index];
            if (gap < 0 || gap > INT64_MAX - weight) {
                return GT_COEFFICIENT_INVALID;
            }
            weight += gap;
        }
        *numerator = weight;
        return GT_COEFFICIENT_OK;
    }
    if (changed == total) {
        return GT_COEFFICIENT_OK;
    }
    return adjacent_coefficient(row, length, changed - start, step, numerator, denominator);
}
