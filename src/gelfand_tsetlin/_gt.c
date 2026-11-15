#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

static PyMethodDef methods[] = {
    {"check_rows", check_rows, METH_O, "Validate top-first rows of exact coordinates."},
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
