#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Each work item owns a mutable word and its exact coefficient. */
static int
push_term(PyObject *work, PyObject *word, PyObject *coefficient)
{
    PyObject *term = PyTuple_Pack(2, word, coefficient);
    if (term == NULL) {
        return -1;
    }
    int status = PyList_Append(work, term);
    Py_DECREF(term);
    return status;
}

static int
push_bracket_term(PyObject *work, PyObject *word, Py_ssize_t position,
                  Py_ssize_t generator, PyObject *coefficient)
{
    Py_ssize_t length = PyList_GET_SIZE(word);
    PyObject *replacement = PyLong_FromSsize_t(generator);
    if (replacement == NULL) {
        return -1;
    }
    PyObject *shorter = PyList_New(length - 1);
    if (shorter == NULL) {
        Py_DECREF(replacement);
        return -1;
    }
    for (Py_ssize_t i = 0; i < length - 1; i++) {
        PyObject *entry = i == position ? replacement :
            Py_NewRef(PyList_GET_ITEM(word, i < position ? i : i + 1));
        PyList_SET_ITEM(shorter, i, entry);
    }
    int status = push_term(work, shorter, coefficient);
    Py_DECREF(shorter);
    return status;
}

static int
accumulate_term(PyObject *result, PyObject *key, PyObject *coefficient)
{
    PyObject *previous = PyDict_GetItemWithError(result, key);
    int status;
    if (previous == NULL) {
        status = PyErr_Occurred() ? -1 : PyDict_SetItem(result, key, coefficient);
    } else {
        PyObject *sum = PyNumber_Add(previous, coefficient);
        if (sum == NULL) {
            status = -1;
        } else {
            int nonzero = PyObject_IsTrue(sum);
            status = nonzero < 0 ? -1 : nonzero ?
                PyDict_SetItem(result, key, sum) : PyDict_DelItem(result, key);
            Py_DECREF(sum);
        }
    }
    return status;
}

/* Continue with the swapped pair in place. Only the shorter bracket terms
 * go on the explicit worklist, so the kernel requires no recursive calls. */
static int
order_word(PyObject *work, PyObject *word, PyObject *coefficient,
           Py_ssize_t rank, PyObject *result)
{
    Py_ssize_t position = 0;
    while (position < PyList_GET_SIZE(word) - 1) {
        PyObject *x = PyList_GET_ITEM(word, position);
        PyObject *y = PyList_GET_ITEM(word, position + 1);
        Py_ssize_t x_id = PyLong_AsSsize_t(x);
        Py_ssize_t y_id = PyLong_AsSsize_t(y);
        if (x_id <= y_id) {
            position++;
            continue;
        }

        /* E_ij E_kl = E_kl E_ij + delta_jk E_il - delta_li E_kj. */
        Py_ssize_t i = x_id / rank, j = x_id % rank;
        Py_ssize_t k = y_id / rank, l = y_id % rank;
        if (j == k && push_bracket_term(
                work, word, position, i * rank + l, coefficient) < 0) {
            return -1;
        }
        if (l == i) {
            PyObject *negative = PyNumber_Negative(coefficient);
            if (negative == NULL) {
                return -1;
            }
            int status = push_bracket_term(work, word, position, k * rank + j, negative);
            Py_DECREF(negative);
            if (status < 0) {
                return -1;
            }
        }
        /* The list already owns one reference to each object. Swap ownership. */
        PyList_SET_ITEM(word, position, y);
        PyList_SET_ITEM(word, position + 1, x);
        if (position > 0) {
            position--;
        }
    }
    PyObject *key = PyList_AsTuple(word);
    if (key == NULL) {
        return -1;
    }
    int status = accumulate_term(result, key, coefficient);
    Py_DECREF(key);
    return status;
}

static int
normalize_product(PyObject *joined, PyObject *coefficient,
                  Py_ssize_t rank, PyObject *result)
{
    PyObject *word = PySequence_List(joined);
    if (word == NULL) {
        return -1;
    }
    PyObject *work = PyList_New(0);
    int status = work == NULL ? -1 : order_word(work, word, coefficient, rank, result);
    Py_DECREF(word);
    while (status == 0 && PyList_GET_SIZE(work) != 0) {
        Py_ssize_t last = PyList_GET_SIZE(work) - 1;
        PyObject *term = Py_NewRef(PyList_GET_ITEM(work, last));
        status = PyList_SetSlice(work, last, last + 1, NULL);
        if (status == 0) {
            status = order_word(
                work, PyTuple_GET_ITEM(term, 0), PyTuple_GET_ITEM(term, 1), rank, result);
        }
        Py_DECREF(term);
    }
    Py_XDECREF(work);
    return status;
}

static int
ordered_boundary(PyObject *left, PyObject *right)
{
    Py_ssize_t left_length = PyTuple_GET_SIZE(left);
    if (left_length == 0 || PyTuple_GET_SIZE(right) == 0) {
        return 1;
    }
    /* Both words were already checked for order and generator range. */
    return PyLong_AsSsize_t(PyTuple_GET_ITEM(left, left_length - 1)) <=
        PyLong_AsSsize_t(PyTuple_GET_ITEM(right, 0));
}

static int
multiply_terms(PyObject *left, PyObject *right, Py_ssize_t rank,
               int canonical, PyObject *result)
{
    PyObject *coefficient = PyNumber_Multiply(
        PyTuple_GET_ITEM(left, 1), PyTuple_GET_ITEM(right, 1));
    if (coefficient == NULL) {
        return -1;
    }
    int status = PyObject_IsTrue(coefficient);
    if (status == 1) {
        PyObject *left_word = PyTuple_GET_ITEM(left, 0);
        PyObject *right_word = PyTuple_GET_ITEM(right, 0);
        PyObject *joined = PySequence_Concat(left_word, right_word);
        if (joined == NULL) {
            status = -1;
        } else if (canonical && ordered_boundary(left_word, right_word)) {
            status = accumulate_term(result, joined, coefficient);
        } else {
            status = normalize_product(joined, coefficient, rank, result);
        }
        Py_XDECREF(joined);
    }
    Py_DECREF(coefficient);
    return status;
}

static int
validate_terms(PyObject *items, Py_ssize_t generators, PyObject *fraction_type,
               int canonical)
{
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(items); i++) {
        PyObject *term = PyList_GET_ITEM(items, i);
        PyObject *word = PyTuple_GET_ITEM(term, 0);
        PyObject *coefficient = PyTuple_GET_ITEM(term, 1);
        if (!PyTuple_CheckExact(word)) {
            PyErr_SetString(PyExc_TypeError, "each PBW word must be a tuple");
            return -1;
        }
        if (!PyLong_CheckExact(coefficient) &&
            (PyObject *)Py_TYPE(coefficient) != fraction_type) {
            PyErr_SetString(PyExc_TypeError, "each PBW coefficient must be an int or Fraction");
            return -1;
        }
        Py_ssize_t previous = -1;
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(word); j++) {
            PyObject *entry = PyTuple_GET_ITEM(word, j);
            if (!PyLong_CheckExact(entry)) {
                PyErr_SetString(PyExc_TypeError, "each PBW generator ID must be an int");
                return -1;
            }
            Py_ssize_t id = PyLong_AsSsize_t(entry);
            if (id < 0 || id >= generators) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError, "PBW generator ID outside the rank");
                }
                return -1;
            }
            if (canonical && id < previous) {
                PyErr_SetString(PyExc_ValueError, "each PBW word must be ordered");
                return -1;
            }
            previous = id;
        }
    }
    return 0;
}

static PyObject *
multiply_pbw_impl(PyObject *module, PyObject *args, int canonical)
{
    Py_ssize_t rank;
    PyObject *left, *right;
    const char *format = canonical ? "nO!O!:multiply_ordered_pbw" : "nO!O!:multiply_pbw";
    if (!PyArg_ParseTuple(args, format, &rank,
                          &PyDict_Type, &left, &PyDict_Type, &right)) {
        return NULL;
    }
    if (rank <= 0 || rank > PY_SSIZE_T_MAX / rank) {
        PyErr_SetString(PyExc_ValueError, "rank must be positive and its square must fit Py_ssize_t");
        return NULL;
    }

    PyObject *fraction_type = *(PyObject **)PyModule_GetState(module);
    PyObject *left_terms = PyDict_Items(left);
    PyObject *right_terms = left_terms == NULL ? NULL : PyDict_Items(right);
    PyObject *result = NULL;
    if (right_terms != NULL &&
        validate_terms(left_terms, rank * rank, fraction_type, canonical) == 0 &&
        validate_terms(right_terms, rank * rank, fraction_type, canonical) == 0) {
        result = PyDict_New();
        if (result != NULL) {
            for (Py_ssize_t i = 0; i < PyList_GET_SIZE(left_terms) && result != NULL; i++) {
                for (Py_ssize_t j = 0; j < PyList_GET_SIZE(right_terms); j++) {
                    if (multiply_terms(PyList_GET_ITEM(left_terms, i),
                                       PyList_GET_ITEM(right_terms, j),
                                       rank, canonical, result) < 0) {
                        Py_CLEAR(result);
                        break;
                    }
                }
            }
        }
    }
    Py_XDECREF(left_terms);
    Py_XDECREF(right_terms);
    return result;
}

static PyObject *
multiply_pbw(PyObject *module, PyObject *args)
{
    return multiply_pbw_impl(module, args, 0);
}

static PyObject *
multiply_ordered_pbw(PyObject *module, PyObject *args)
{
    return multiply_pbw_impl(module, args, 1);
}

static int
uea_traverse(PyObject *module, visitproc visit, void *arg)
{
    Py_VISIT(*(PyObject **)PyModule_GetState(module));
    return 0;
}

static int
uea_clear(PyObject *module)
{
    Py_CLEAR(*(PyObject **)PyModule_GetState(module));
    return 0;
}

static PyMethodDef methods[] = {
    {"multiply_pbw", multiply_pbw, METH_VARARGS, "Multiply and normal-order PBW terms."},
    {"multiply_ordered_pbw", multiply_ordered_pbw, METH_VARARGS,
     "Multiply PBW polynomials whose words are already ordered."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef definition = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_uea",
    .m_size = sizeof(PyObject *),
    .m_methods = methods,
    .m_traverse = uea_traverse,
    .m_clear = uea_clear,
};

PyMODINIT_FUNC
PyInit__uea(void)
{
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) {
        return NULL;
    }
    PyObject *fractions = PyImport_ImportModule("fractions");
    if (fractions == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    PyObject **fraction_type = PyModule_GetState(module);
    *fraction_type = PyObject_GetAttrString(fractions, "Fraction");
    Py_DECREF(fractions);
    if (*fraction_type == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
