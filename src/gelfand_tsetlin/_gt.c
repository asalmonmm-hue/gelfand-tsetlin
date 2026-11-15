#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "gt_coefficient.h"
#include "gt_inner_product.h"

/* Results use the Python C API convention: 1 true, 0 false, -1 error. */
static int
is_nonnegative_integer_gap(PyObject *upper, PyObject *lower)
{
    PyObject *upper_real = PyTuple_GET_ITEM(upper, 0);
    PyObject *lower_real = PyTuple_GET_ITEM(lower, 0);
    int equal_imaginary = PyObject_RichCompareBool(
        PyTuple_GET_ITEM(upper, 1), PyTuple_GET_ITEM(lower, 1), Py_EQ);
    if (equal_imaginary <= 0) {
        return equal_imaginary;
    }

    int ordered = PyObject_RichCompareBool(upper_real, lower_real, Py_GE);
    if (ordered <= 0 || (PyLong_Check(upper_real) && PyLong_Check(lower_real))) {
        return ordered;
    }

    /* The frontend marshals real parts as exact integers or fractions. */
    PyObject *gap = PyNumber_Subtract(upper_real, lower_real);
    if (gap == NULL) {
        return -1;
    }
    PyObject *integer = PyNumber_Long(gap);
    if (integer == NULL) {
        Py_DECREF(gap);
        return -1;
    }
    int integral = PyObject_RichCompareBool(gap, integer, Py_EQ);
    Py_DECREF(integer);
    Py_DECREF(gap);
    return integral;
}

static PyObject *
check_rows(PyObject *Py_UNUSED(self), PyObject *rows)
{
    if (!PyTuple_Check(rows)) {
        Py_RETURN_FALSE;
    }
    Py_ssize_t rank = PyTuple_GET_SIZE(rows);
    if (rank == 0) {
        Py_RETURN_FALSE;
    }

    PyObject *upper = NULL;
    for (Py_ssize_t i = 0; i < rank; i++) {
        PyObject *row = PyTuple_GET_ITEM(rows, i);
        Py_ssize_t length = rank - i;
        if (!PyTuple_Check(row) || PyTuple_GET_SIZE(row) != length) {
            Py_RETURN_FALSE;
        }
        for (Py_ssize_t j = 0; j < length; j++) {
            PyObject *entry = PyTuple_GET_ITEM(row, j);
            if (!PyTuple_Check(entry) || PyTuple_GET_SIZE(entry) != 2) {
                Py_RETURN_FALSE;
            }
            if (upper == NULL) {
                continue;
            }
            int valid = is_nonnegative_integer_gap(PyTuple_GET_ITEM(upper, j), entry);
            if (valid == 1) {
                valid = is_nonnegative_integer_gap(entry, PyTuple_GET_ITEM(upper, j + 1));
            }
            if (valid < 0) {
                return NULL;
            }
            if (valid == 0) {
                Py_RETURN_FALSE;
            }
        }
        upper = row;
    }
    Py_RETURN_TRUE;
}

static PyObject *
check_integer_rows(PyObject *Py_UNUSED(self), PyObject *rows)
{
    if (!PyList_CheckExact(rows) && !PyTuple_CheckExact(rows)) {
        Py_RETURN_NONE;
    }
    Py_ssize_t rank = PySequence_Fast_GET_SIZE(rows);
    PyObject *const *upper = NULL;
    int valid = rank > 0;
    for (Py_ssize_t i = 0; i < rank; i++) {
        PyObject *row = PySequence_Fast_GET_ITEM(rows, i);
        if (!PyList_CheckExact(row) && !PyTuple_CheckExact(row)) {
            Py_RETURN_NONE;
        }
        Py_ssize_t length = PySequence_Fast_GET_SIZE(row);
        PyObject *const *entries = PySequence_Fast_ITEMS(row);
        if (length != rank - i) {
            valid = 0;
        }
        /* Continue guarding types after failure so fallback still handles any
         * custom iteration or scalar conversion elsewhere in the input. */
        for (Py_ssize_t j = 0; j < length; j++) {
            PyObject *entry = entries[j];
            if (!PyLong_CheckExact(entry)) {
                Py_RETURN_NONE;
            }
            if (valid && upper != NULL) {
                valid = PyObject_RichCompareBool(upper[j], entry, Py_GE);
                if (valid == 1) {
                    valid = PyObject_RichCompareBool(entry, upper[j + 1], Py_GE);
                }
                if (valid < 0) {
                    return NULL;
                }
            }
        }
        upper = entries;
    }
    return PyBool_FromLong(valid);
}

/* This bridge owns the input buffers. The kernels only borrow them. */
static int64_t *
pack_patterns(Py_ssize_t rank, PyObject *source, PyObject *target)
{
    size_t count = (size_t)PyTuple_GET_SIZE(source);
    size_t sides = target == NULL ? 1 : 2;
    size_t n = (size_t)rank;
    size_t a = n % 2 ? n : n / 2;
    size_t b = n % 2 ? n / 2 + 1 : n + 1;
    if (rank <= 0 || b > count / a || a * b != count ||
        (target != NULL && count != (size_t)PyTuple_GET_SIZE(target)) ||
        count > (size_t)PY_SSIZE_T_MAX / (sides * sizeof(int64_t))) {
        PyErr_SetString(PyExc_ValueError, "invalid native pattern buffer size");
        return NULL;
    }

    int64_t *packed = PyMem_New(int64_t, sides * count);
    if (packed == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    PyObject *patterns[] = {source, target};
    for (size_t side = 0; side < sides; side++) {
        for (size_t k = 0; k < count; k++) {
            long long value = PyLong_AsLongLong(PyTuple_GET_ITEM(patterns[side], k));
            if (value < 0 || value > INT64_MAX) {
                PyMem_Free(packed);
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_OverflowError, "pattern offset outside int64 range");
                }
                return NULL;
            }
            packed[side * count + k] = (int64_t)value;
        }
    }
    return packed;
}

static int
check_status(enum gt_coefficient_status status)
{
    if (status == GT_COEFFICIENT_OVERFLOW) {
        PyErr_SetString(PyExc_OverflowError, "GT arithmetic exceeded its fixed-width range");
        return 0;
    }
    if (status != GT_COEFFICIENT_OK) {
        PyErr_SetString(PyExc_ValueError, "invalid GT kernel arguments");
        return 0;
    }
    return 1;
}

static PyObject *
fraction_result(enum gt_coefficient_status status, int64_t numerator, int64_t denominator)
{
    if (!check_status(status)) {
        return NULL;
    }
    return Py_BuildValue("(LL)", (long long)numerator, (long long)denominator);
}

static PyObject *
coefficient_i64(PyObject *Py_UNUSED(self), PyObject *args)
{
    Py_ssize_t rank, i, j;
    PyObject *source, *target;
    if (!PyArg_ParseTuple(args, "nnnO!O!:coefficient_i64", &rank, &i, &j,
                          &PyTuple_Type, &source, &PyTuple_Type, &target)) {
        return NULL;
    }
    int64_t *packed = pack_patterns(rank, source, target);
    if (packed == NULL) {
        return NULL;
    }
    int64_t numerator = 0, denominator = 1;
    enum gt_coefficient_status status = gt_coefficient(
        (size_t)rank, (size_t)i, (size_t)j, packed, packed + PyTuple_GET_SIZE(source),
        &numerator, &denominator);
    PyMem_Free(packed);
    return fraction_result(status, numerator, denominator);
}

static PyObject *
adjacent_coefficients_i64(PyObject *Py_UNUSED(self), PyObject *args)
{
    Py_ssize_t rank, i, j;
    PyObject *source;
    if (!PyArg_ParseTuple(args, "nnnO!:adjacent_coefficients_i64", &rank, &i, &j,
                          &PyTuple_Type, &source)) {
        return NULL;
    }
    Py_ssize_t length = i < j ? i : j;
    Py_ssize_t larger = i < j ? j : i;
    if (length <= 0 || larger > rank || larger - length != 1) {
        PyErr_SetString(PyExc_ValueError, "indices must be adjacent and within the pattern rank");
        return NULL;
    }
    int64_t *packed = pack_patterns(rank, source, NULL);
    if (packed == NULL) {
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (result != NULL) {
        for (Py_ssize_t column = 0; column < length; column++) {
            int64_t numerator, denominator;
            enum gt_coefficient_status status = gt_adjacent_coefficient(
                (size_t)rank, (size_t)i, (size_t)j, packed, (size_t)column,
                &numerator, &denominator);
            if (!check_status(status)) {
                Py_CLEAR(result);
                break;
            }
            if (numerator != 0) {
                PyObject *term = Py_BuildValue("(nLL)", column, (long long)numerator,
                                              (long long)denominator);
                int appended = term == NULL ? -1 : PyList_Append(result, term);
                Py_XDECREF(term);
                if (appended < 0) {
                    Py_CLEAR(result);
                    break;
                }
            }
        }
    }
    PyMem_Free(packed);
    return result;
}

static PyObject *
inner_product_i64(PyObject *Py_UNUSED(self), PyObject *args)
{
    Py_ssize_t rank;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "nO!O!:inner_product_i64", &rank,
                          &PyTuple_Type, &left, &PyTuple_Type, &right)) {
        return NULL;
    }
    int64_t *packed = pack_patterns(rank, left, right);
    if (packed == NULL) {
        return NULL;
    }
    int64_t numerator = 0, denominator = 1;
    enum gt_coefficient_status status = gt_inner_product(
        (size_t)rank, packed, packed + PyTuple_GET_SIZE(left), &numerator, &denominator);
    PyMem_Free(packed);
    return fraction_result(status, numerator, denominator);
}

static PyObject *
shifted_integer(PyObject *entry, Py_ssize_t column)
{
    if (!PyLong_CheckExact(entry)) {
        PyErr_SetString(PyExc_TypeError, "highest-line entries must be exact integers");
        return NULL;
    }
    if (column == 0) {
        return Py_NewRef(entry);
    }
    PyObject *offset = PyLong_FromSsize_t(column);
    if (offset == NULL) {
        return NULL;
    }
    PyObject *shifted = PyNumber_Subtract(entry, offset);
    Py_DECREF(offset);
    return shifted;
}

static PyObject *
highest_line_coefficient(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *top, *middle;
    Py_ssize_t i;
    int step;
    if (!PyArg_ParseTuple(args, "O!O!ni:highest_line_coefficient",
                          &PyTuple_Type, &top, &PyTuple_Type, &middle, &i, &step)) {
        return NULL;
    }
    Py_ssize_t rank = PyTuple_GET_SIZE(top);
    if (rank < 2 || PyTuple_GET_SIZE(middle) != rank - 1 ||
        i < 1 || i >= rank || (step != 1 && step != -1)) {
        PyErr_SetString(PyExc_ValueError, "invalid highest-line rows, index, or step");
        return NULL;
    }

    PyObject *entry = PyTuple_GET_ITEM(middle, i - 1);
    PyObject *bound = PyTuple_GET_ITEM(top, step == 1 ? i - 1 : i);
    if (!PyLong_CheckExact(entry) || !PyLong_CheckExact(bound)) {
        PyErr_SetString(PyExc_TypeError, "highest-line entries must be exact integers");
        return NULL;
    }
    int admissible = PyObject_RichCompareBool(entry, bound, step == 1 ? Py_LT : Py_GT);
    if (admissible < 0) {
        return NULL;
    }
    if (!admissible || step == -1) {
        return PyLong_FromLong(admissible);
    }

    /* Molev (2.28), evaluated with exact integers of arbitrary size. */
    PyObject *m = shifted_integer(entry, i - 1);
    if (m == NULL) {
        return NULL;
    }
    PyObject *product = PyLong_FromLong(-1);
    for (Py_ssize_t j = 0; product != NULL && j < rank; j++) {
        PyObject *ell = shifted_integer(PyTuple_GET_ITEM(top, j), j);
        PyObject *factor = ell == NULL ? NULL : PyNumber_Subtract(m, ell);
        Py_XDECREF(ell);
        PyObject *next = factor == NULL ? NULL : PyNumber_Multiply(product, factor);
        Py_XDECREF(factor);
        Py_DECREF(product);
        product = next;
    }
    Py_DECREF(m);
    return product;
}

static PyMethodDef methods[] = {
    {"check_rows", check_rows, METH_O, "Validate top-first rows of exact coordinates."},
    {"check_integer_rows", check_integer_rows, METH_O,
     "Validate builtin integer rows or return None for frontend fallback."},
    {"coefficient_i64", coefficient_i64, METH_VARARGS, "Evaluate a native GT coefficient."},
    {"adjacent_coefficients_i64", adjacent_coefficients_i64, METH_VARARGS,
     "Evaluate all nonzero terms for one adjacent generator."},
    {"inner_product_i64", inner_product_i64, METH_VARARGS, "Evaluate the native GT pairing."},
    {"highest_line_coefficient", highest_line_coefficient, METH_VARARGS,
     "Evaluate the exact raising or lowering coefficient between highest lines."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_gt",
    .m_size = 0,
    .m_methods = methods,
};

PyMODINIT_FUNC
PyInit__gt(void)
{
    return PyModule_Create(&module);
}
